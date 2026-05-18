# image_features.py
from __future__ import annotations
import os, cv2, numpy as np, hashlib
from typing import Optional, Tuple
from PIL import Image
from app.delta_comparator.utils.logger import log as logging

# pytesseract is optional; log clearly if binary missing
try:
    import pytesseract
    HAVE_TESS = True
except Exception as e:
    HAVE_TESS = False
    logging.warning(f"pytesseract import failed: {e}", )

# CLIP toggle via env
_CLIP = None
_PREPROCESS = None
_DISABLE_CLIP = os.environ.get("DISABLE_CLIP", "0") == "1"

def _load_clip():
    global _CLIP, _PREPROCESS
    if _DISABLE_CLIP:
        #logging.debug("CLIP disabled via DISABLE_CLIP=1")
        return None, None
    if _CLIP is not None:
        return _CLIP, _PREPROCESS
    try:
        import open_clip, torch
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        _CLIP, _PREPROCESS = (model, (preprocess, device))
        #logging.debug(f"Loaded CLIP ViT-B-32 (device={device})")
        return _CLIP, _PREPROCESS
    except Exception as e:
        logging.warning(f"CLIP load failed (will proceed without it): {e}")
        return None, None

def clip_image_embedding(img_path: str) -> Optional[np.ndarray]:
    try:
        model, pp = _load_clip()
        if model is None or pp is None:
            return None
        preprocess, device = pp
        from PIL import Image as PILImage
        im = PILImage.open(img_path).convert("RGB")
        import torch
        with torch.no_grad():
            t = preprocess(im).unsqueeze(0).to(device)
            z = model.encode_image(t)
            z = z / z.norm(dim=-1, keepdim=True)
        vec = z.squeeze(0).detach().cpu().numpy().astype("float32")
        return vec
    except Exception as e:
        #logging.debug(f"clip_image_embedding failed on {img_path}: {e}")
        return None

def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None: return 0.0
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

# perceptual hashes
def _img_for_hash(path, size=(256,256)):
    try:
        im = Image.open(path).convert("L").resize(size, Image.BILINEAR)
        return np.array(im)
    except Exception as e:
        #logging.debug(f"_img_for_hash failed on {path}: {e}")
        return None

def _phash(arr: np.ndarray) -> str:
    arr = cv2.dct(arr.astype(np.float32))
    block = arr[:8, :8]
    med = np.median(block)
    bits = (block > med).flatten().astype(np.uint8)
    return "".join("1" if b else "0" for b in bits)

def _dhash(arr: np.ndarray) -> str:
    arr = cv2.resize(arr, (9, 8), interpolation=cv2.INTER_LINEAR)
    diff = arr[:, 1:] > arr[:, :-1]
    return "".join("1" if b else "0" for b in diff.flatten())

def hamming(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b): return 64
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

def image_hashes(img_path: str) -> Tuple[Optional[str], Optional[str]]:
    arr = _img_for_hash(img_path)
    if arr is None: return None, None
    try:
        ph = _phash(arr); dh = _dhash(arr)
        return ph, dh
    except Exception as e:
        #logging.debug(f"image_hashes failed on {img_path}: {e}")
        return None, None

def hash_similarity(ph_a: Optional[str], ph_b: Optional[str],
                    dh_a: Optional[str]=None, dh_b: Optional[str]=None) -> Tuple[float, float]:
    ph = 0.0
    if ph_a and ph_b:
        ph = 1.0 - (hamming(ph_a, ph_b) / 64.0)
    dh = 0.0
    if dh_a and dh_b:
        dh = 1.0 - (hamming(dh_a, dh_b) / 64.0)
    return ph, dh

# ORB inliers
def orb_inlier_score(path_a: str, path_b: str) -> float:
    try:
        img1 = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None: return 0.0
        orb = cv2.ORB_create(1000)
        k1, d1 = orb.detectAndCompute(img1, None)
        k2, d2 = orb.detectAndCompute(img2, None)
        if d1 is None or d2 is None: return 0.0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d1, d2)
        if not matches: return 0.0
        matches = sorted(matches, key=lambda x: x.distance)[:200]
        pts1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if mask is None: return 0.0
        inliers = float(mask.sum())
        return inliers / max(1.0, float(len(matches)))
    except Exception as e:
        #logging.debug(f"orb_inlier_score failed on {path_a} vs {path_b}: {e}")
        return 0.0

# OCR
def ocr_text(img_path: str) -> str:
    if not HAVE_TESS:
        logging.warning("pytesseract not available (no OCR). "
                       "Install pytesseract and the Tesseract binary (set PATH or pytesseract.pytesseract.tesseract_cmd)")
        return ""
    try:
        im = Image.open(img_path).convert("RGB")
        text = pytesseract.image_to_string(im, lang="eng")
        text = " ".join(text.split())
        #logging.debug(f"OCR [{os.path.basename(img_path)}] -> {len(text)} chars | sample='{text[:120]}...'")
        return text
    except Exception as e:
        logging.debug(f"OCR failed on {img_path}: {e}")
        return ""

def vision_caption(img_path: str) -> str:
    # keep as no-op unless you wire Azure Vision; add logs if you enable
    return ""
