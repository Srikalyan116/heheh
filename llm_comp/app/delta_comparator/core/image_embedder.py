# image_embedder.py  (patched drop-in)
import os
import json
import numpy as np
from typing import Optional, Tuple, List
from PIL import Image
import platform
from pathlib import Path
from collections import Counter
from app.delta_comparator.utils.logger import log as logging
import onnxruntime as ort

try:
    import open_clip
    import torch
    _HAVE_CLIP = True
except Exception:
    open_clip = None
    torch = None
    _HAVE_CLIP = False

# Try import cv2 for color ops; optional but recommended
try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False
    cv2 = None


_EMBEDDER_CACHE = {}


def _normalize_device(device: Optional[str]) -> str:
    requested = (device or "cpu").strip().lower()
    if requested in {"gpu", "cuda"}:
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return "cpu"

def _tiny_color_hist(img: Image.Image, bins: int = 32) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"))
    r = np.histogram(arr[...,0].ravel(), bins=bins, range=(0,255))[0]
    g = np.histogram(arr[...,1].ravel(), bins=bins, range=(0,255))[0]
    b = np.histogram(arr[...,2].ravel(), bins=bins, range=(0,255))[0]
    h = np.concatenate([r,g,b]).astype(np.float32)
    n = np.linalg.norm(h) or 1.0
    return h / n

def dominant_colors(img: Image.Image, k: int = 4, sample_limit: int = 100000) -> List[Tuple[int,int,int]]:
    arr = np.asarray(img.convert("RGB")).reshape(-1, 3).astype(np.float32)
    if arr.shape[0] > sample_limit:
        step = max(1, arr.shape[0] // sample_limit)
        arr_sample = arr[::step][:sample_limit]
    else:
        arr_sample = arr
    if _HAVE_CV2:
        Z = arr_sample.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
        K = min(k, max(1, int(len(Z)/50)))
        if K <= 0:
            return []
        try:
            _, labels, centers = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
            centers = centers.astype(int).tolist()
            return [tuple(map(int, c)) for c in centers]
        except Exception:
            pass
    # fallback histogram sampling
    arr_uint8 = arr_sample.astype(np.uint8)
    flat = [tuple(c.tolist()) for c in arr_uint8]
    cnt = Counter(flat)
    most = [c for c,_ in cnt.most_common(k)]
    return most


# -------------------------
# CLIP ONNX embedder wrapper
# -------------------------
class _ClipEmbedderONNX:
    """
    OpenCLIP vision embedder using ONNX Runtime.
    Falls back to histogram if anything fails.
    """

    def __init__(self, onnx_dir: str, device: str = "cpu"):
        self.onnx_dir = onnx_dir
        self.device = device
        self.session = None
        self.preprocess = None
        self._load()

    def _load(self):
        try:
            # model_path = os.path.join(self.onnx_dir, "model.onnx")
            # if not os.path.exists(model_path):
            #     logging.warning("[ClipEmbedderONNX] model.onnx not found. Falling back.")
            #     return

            # providers = ["CPUExecutionProvider"]
            # self.session = ort.InferenceSession(model_path, providers=providers)

            # # IMPORTANT: preprocessing must match OpenCLIP
            # _, preprocess, _ = open_clip.create_model_and_transforms(
            #     "ViT-B-32",
            #     pretrained="laion2b_s34b_b79k"
            # )
            # self.preprocess = preprocess

            # logging.info(f"[ClipEmbedderONNX] Loaded ONNX model from {self.onnx_dir}")
            model_path = os.path.join(self.onnx_dir, "model.onnx")

            providers = ["CPUExecutionProvider"]
            use_cuda = str(os.environ.get("USE_ONNX_CUDA", "0")).strip().lower() in {"1", "true", "yes", "on"}
            if self.device == "cuda" or use_cuda:
                available = set(ort.get_available_providers())
                if "CUDAExecutionProvider" in available:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            self.session = ort.InferenceSession(model_path, providers=providers)

            from torchvision import transforms
            from PIL import Image

            self.preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)
                )
            ])

            #logging.debug(f"[ClipEmbedderONNX] Loaded ONNX model from {self.onnx_dir}")

        except Exception as e:
            logging.error(f"[ClipEmbedderONNX] Load failed: {e}")
            self.session = None
            self.preprocess = None

    def embed(self, img: Image.Image) -> np.ndarray:
        if self.session is None or self.preprocess is None:
            return _tiny_color_hist(img)

        try:
            pixel_values = self.preprocess(img).unsqueeze(0).numpy()
            outputs = self.session.run(None, {"pixel_values": pixel_values})

            emb = outputs[0][0]          # (512,)
            emb = emb / (np.linalg.norm(emb) or 1.0)
            return emb.astype(np.float32)

        except Exception as e:
            logging.error(f"[ClipEmbedderONNX] embed failed: {e}")
            return _tiny_color_hist(img)

# -------------------------
# CLIP embedder wrapper
# -------------------------
class _ClipEmbedder:
    """
    Load CLIP model from a local directory if available (checkpoint.pt + meta.json).
    If loading fails, .model will be None and embed() falls back to color histogram.
    """
    def __init__(self, local_dir: str, device: str = "cpu"):
        self.local_dir = str(local_dir) if local_dir is not None else None
        self.device = device
        self.model = None
        self.preprocess = None
        self._load_from_local()

    @staticmethod
    def _maybe_path(p: str) -> Optional[str]:
        """Return a usable path string or None. Try normal and Windows \\?\\ prefix."""
        if not p:
            return None
        if os.path.exists(p):
            return p
        # try Windows extended prefix
        if platform.system().lower().startswith("win"):
            try:
                p2 = r"\\?\\" + p
                if os.path.exists(p2):
                    return p2
            except Exception:
                pass
        return None

    def _load_from_local(self):
        if not _HAVE_CLIP:
            logging.warning("[ClipEmbedder] open_clip not installed. Using fallback embedding.")
            return

        if not self.local_dir:
            logging.warning("[ClipEmbedder] local_dir not provided. Using fallback embedding.")
            return

        # resolve candidate dir
        cand_dir = os.path.abspath(self.local_dir)
        cand_ok = self._maybe_path(cand_dir)
        if not cand_ok:
            # diagnostic: print ancestor existence map
            path_parts = Path(cand_dir).parts
            accum = Path(path_parts[0])
            #logging.debug(f"[ClipEmbedder] local_dir not accessible: {cand_dir!r}")
            #logging.debug("[ClipEmbedder] ancestor existence (first missing stops):")
            try:
                accum = Path(path_parts[0])
            except Exception:
                accum = Path(cand_dir)
            for i in range(1, len(path_parts)+1):
                p = Path("").joinpath(*path_parts[:i])
                ex = os.path.exists(str(p))
                #logging.debug(f"  {str(p)} -> exists={ex}")
                if not ex:
                    break
            #logging.debug("[ClipEmbedder] Falling back to histogram embedder.")
            return

        # use the working path that exists
        model_dir = cand_ok
        meta_path = os.path.join(model_dir, "meta.json")
        ckpt_path = os.path.join(model_dir, "checkpoint.pt")
        meta_ok = self._maybe_path(meta_path)
        ckpt_ok = self._maybe_path(ckpt_path)
        if not meta_ok:
            logging.warning(f"[ClipEmbedder] meta.json missing in {model_dir} (checked robustly). Falling back.")
            return
        if not ckpt_ok:
            logging.warning(f"[ClipEmbedder] checkpoint.pt missing in {model_dir} (checked robustly). Falling back.")
            return

        # load metadata
        try:
            with open(meta_ok, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            logging.error(f"[ClipEmbedder] failed to read meta.json: {e}. Falling back.")
            return

        model_name = meta.get("model_name")
        if not model_name:
            logging.warning(f"[ClipEmbedder] meta.json missing 'model_name' entry. Falling back.")
            return

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=None,
                device=self.device
            )
        except Exception as e:
            logging.error(f"[ClipEmbedder] open_clip.create_model_and_transforms failed: {e}. Falling back.")
            return

        try:
            sd = torch.load(ckpt_ok, map_location=self.device)
            if "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            #model.load_state_dict(sd, strict=False)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            model.eval()
            self.model = model
            self.preprocess = preprocess
            logging.info(f"[ClipEmbedder] Loaded CLIP model from: {model_dir}")
            
            logging.info(
                "[ClipEmbedder] CLIP checkpoint loaded successfully | "
                f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
            )

        except Exception as e:
            logging.error(f"[ClipEmbedder] Error loading checkpoint: {e}. Falling back to histogram embedder.")
            self.model = None
            self.preprocess = None

    def embed(self, img: Image.Image) -> np.ndarray:
        if self.model is None or self.preprocess is None:
            return _tiny_color_hist(img)
        try:
            import torch
            with torch.no_grad():
                t = self.preprocess(img).unsqueeze(0).to(self.device)
                feats = self.model.encode_image(t)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                return feats[0].cpu().numpy().astype(np.float32)
        except Exception as e:
            logging.error(f"[ClipEmbedder] embed failed: {e}. Falling back to histogram.")
            return _tiny_color_hist(img)

# Factory with env var override and robust checks
# def get_image_embedder(local_dir: Optional[str] = None, device: str = "cpu"):
#     """
#     Prefer explicit env var OPENCLIP_MODEL_DIR if set.
#     Falls back to repo-relative models location.
#     If the target directory cannot be seen from this process, the returned _ClipEmbedder
#     will gracefully fall back to a histogram embedder (model=None).
#     """
#     env_dir = os.environ.get("OPENCLIP_MODEL_DIR")
#     if env_dir:
#         logging.info("Openclip loaded via .env")
#         use_dir = env_dir
#     elif local_dir:
#         use_dir = local_dir
#         logging.info("Openclip loaded via local directory")
#     else:
#         here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         #use_dir = os.path.join(here, "models", "openclip", "ViT-B-32__laion2b_s34b_b79k")
#         use_dir = os.path.join(here, "models", "openclip")
#         logging.info(f"local model dir path: {use_dir}")
#         logging.info("Openclip loaded via project directory")
#     # return an embedder instance (itself checks paths and falls back)
#     return _ClipEmbedder(local_dir=use_dir, device=device)

def get_image_embedder(
    local_dir: Optional[str] = None,
    device: str = "cpu"
):
    """
    Environment-driven OpenCLIP loader.
    Priority:
    1. USE_OPENCLIP_ONNX=1 -> ONNX embedder
    2. OPENCLIP_MODEL_DIR -> PyTorch embedder
    3. Fallback -> color histogram
    """
    normalized_device = "cpu" #_normalize_device(device)
    use_onnx = os.environ.get("USE_OPENCLIP_ONNX")

    if use_onnx:
        onnx_dir = os.environ.get("OPENCLIP_ONNX_DIR")
        if onnx_dir:
            cache_key = ("onnx", os.path.abspath(onnx_dir), normalized_device)
            cached = _EMBEDDER_CACHE.get(cache_key)
            if cached is not None:
                return cached
            logging.info(f"[ImageEmbedder ONNX] Using OpenCLIP ONNX backend on {normalized_device}")
            embedder = _ClipEmbedderONNX(onnx_dir=onnx_dir, device=normalized_device)
            _EMBEDDER_CACHE[cache_key] = embedder
            return embedder
        else:
            logging.warning(
                "[ImageEmbedder] USE_OPENCLIP_ONNX=1 but OPENCLIP_ONNX_DIR not set"
            )

    # # ---- fallback to PyTorch OpenCLIP ----
    # env_dir = os.environ.get("OPENCLIP_MODEL_DIR")
                                                                                                
    # if env_dir:
    #     logging.info("[ImageEmbedder Torch] Using OpenCLIP PyTorch backend (env)")
    #     return _ClipEmbedder(local_dir=env_dir, device=device)

    # if local_dir:
    #     logging.info("[ImageEmbedder Torch Local] Using OpenCLIP PyTorch backend (local)")
    #     return _ClipEmbedder(local_dir=local_dir, device=device)

    cache_key = ("fallback", os.path.abspath(local_dir) if local_dir else None, normalized_device)
    cached = _EMBEDDER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    logging.warning("[ImageEmbedder] No OpenCLIP backend available. Using histogram.")
    embedder = _ClipEmbedder(local_dir=None, device=normalized_device)
    _EMBEDDER_CACHE[cache_key] = embedder
    return embedder