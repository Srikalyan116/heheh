# image_prepass_matcher.py
from __future__ import annotations
import os, re, math, unicodedata, io, json, time
from contextlib import suppress
from typing import Any, Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

import torch
from app.delta_comparator.utils.logger import log as logging
from rabbitMQ_manager import create_queue_publish
import numpy as np
from PIL import Image
from app.delta_comparator.core.onnx_sbert_loader import get_sbert_model

# --- Optional libs (graceful fallbacks) ---
try:
    import pytesseract
    _HAVE_TESS = True
except Exception:
    pytesseract = None
    _HAVE_TESS = False

# try:
#     from sentence_transformers import SentenceTransformer
#     _HAVE_SBERT = True
# except Exception:
#     SentenceTransformer = None
#     _HAVE_SBERT = False

# image embedder should implement embed(Image) returning numpy vector
from app.delta_comparator.core.image_embedder import get_image_embedder, _tiny_color_hist, dominant_colors

try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    cv2 = None
    _HAVE_CV2 = False

try:
    from rapidfuzz import fuzz, process as rproc
    _HAVE_RAPIDFUZZ = True
except Exception:
    _HAVE_RAPIDFUZZ = False

import difflib

# ---------------------------
# Global tunables (change as needed)
# # ---------------------------
# GLOBAL_IMG_WEIGHT = 0.6
# GLOBAL_TEXT_WEIGHT = 0.4
GLOBAL_IMG_WEIGHT = 0.5
GLOBAL_TEXT_WEIGHT = 0.5
# =========================
# Utilities
# =========================

_MOJIBAKE_FIXES = {
    "Â°": "°", "â‰¤": "≤", "â‰¥": "≥", "â€“": "–", "â€”": "—",
    "â€¢": "•", "â€˜": "‘", "â€™": "’", "â€œ": "“", "â€": "”",
    "â€¦": "…", "Â±": "±", "Â·": "·", "Â®": "®", "Â©": "©",
}

_MD_MARKS = re.compile(r"[*_`]+")
_BULLETS  = re.compile(r"[•·●◦]+")
_SPACES   = re.compile(r"\s+")
_SIDE_NOTES = re.compile(
    r"\b(if\s+present|depending\s+on\s+vehicle\s+variant|case\s+\d+[:]?|example\s+of\s+intervention\s+profile)\b",
    re.I,
)
_CODE_FENCE = re.compile(r"```+[\w\-]*", re.I)
_MARKDOWN_WORD = re.compile(r"\bmarkdown\b", re.I)

def _fix_mojibake(s: str) -> str:
    if not s: return ""
    t = s
    for bad, good in _MOJIBAKE_FIXES.items():
        t = t.replace(bad, good)
    return unicodedata.normalize("NFC", t)

def _normalize_ocr(txt: str) -> str:
    if not txt:
        return ""
    t = _fix_mojibake(txt)
    t = _CODE_FENCE.sub(" ", t)              # remove ``` or ```markdown
    t = _MARKDOWN_WORD.sub(" ", t)           # remove literal 'markdown'
    t = t.replace("&amp;", "&")
    t = _MD_MARKS.sub("", t)
    t = _BULLETS.sub(" ", t)
    t = t.replace(" + ", " & ").replace("+", " & ")
    t = re.sub(r"[|]+", " ", t)
    t = re.sub(r"[–—\-]{2,}", " - ", t)
    t = _SIDE_NOTES.sub(" ", t)
    t = _SPACES.sub(" ", t).strip()
    return t

def _ocr_text(img: Image.Image) -> str:
    if not _HAVE_TESS:
        return ""
    try:
        return pytesseract.image_to_string(img, config="--psm 6").strip()
    except Exception:
        return ""

def _summarize_ocr(txt: str) -> str:
    lines = [ln.strip() for ln in (txt or "").splitlines() if ln.strip()]
    if not lines: return ""
    head = lines[0]
    caps = re.findall(r"\b[A-Z]{2,}\b", txt or "")
    caps = sorted(set(caps))[:8]
    if caps:
        head = f"{head} — shows: {', '.join(caps)}"
    return head[:240]

def _clean_line_basic(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\((?:[^)]*)\)", " ", s)
    s = _SPACES.sub(" ", s).strip()
    return s

_CONNECT_SPLIT = re.compile(r"\s*(?:&|/|\+|,|;|\band\b)\s*", re.I)

def _explode_composites(line: str) -> List[str]:
    base = _clean_line_basic(line)
    parts = [p.strip() for p in _CONNECT_SPLIT.split(base) if p.strip()]
    out = set(parts)
    if 2 <= len(parts) <= 3:
        out.add(" ".join(parts))
    return sorted(out)

def _line_bag(txt: str) -> List[str]:
    raw = [ln for ln in (txt or "").splitlines()]
    bag = set()
    for ln in raw:
        ln = ln.strip()
        if not ln: continue
        bag.add(_clean_line_basic(ln))
        for atom in _explode_composites(ln):
            bag.add(atom)
    bag = {b for b in bag if len(b) >= 2}
    return sorted(bag)

def _entities(txt: str) -> List[str]:
    if not txt:
        return []
    ents = re.findall(r"\b[A-Z][A-Z0-9]{1,6}\b", txt)
    ents = [e.replace("_", "").replace("-", "") for e in ents]
    return sorted(set(ents))

def _atoms_from_entities_and_pairs(txt: str) -> List[str]:
    atoms = set()
    for ln in (txt or "").splitlines():
        c = [x for x in re.findall(r"\b[A-Z][A-Z0-9]{1,6}\b", ln)]
        if len(c) >= 2:
            for i in range(len(c)):
                atoms.add(c[i].lower())
                for j in range(i+1, len(c)):
                    atoms.add((c[i] + " " + c[j]).lower())
        elif len(c) == 1:
            atoms.add(c[0].lower())
    return sorted(atoms)

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb: return 0.0
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union

# SBERT loader
_SBERT = None
# def _get_sbert_local():
#     global _SBERT
#     if _SBERT is not None:
#         return _SBERT
#     if not _HAVE_SBERT:
#         return None
#     here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     local_dir = os.path.join(here, "models", "all-MiniLM-L6-v2")
#     logging.info(f"local model dir path: {local_dir}")
#     try:
#         _SBERT = SentenceTransformer(local_dir)
#         logging.info(f"[SBERT] Loaded local SBERT model from {local_dir}.")
#         return _SBERT
#     except Exception:
#         pass
#     try:
#         _SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#         logging.info(f"[SBERT] Loaded SBERT model from HuggingFace hub.")
#         return _SBERT
#     except Exception:
#         return None

def _hash_vec(s: str, dim: int = 256) -> np.ndarray:
    toks = re.findall(r"\w+", (s or "").lower())
    v = np.zeros(dim, dtype=np.float32)
    for t in toks:
        v[hash(t) % dim] += 1.0
    n = np.linalg.norm(v) or 1.0
    return v / n

# =========================
# Robust lineification helpers
# =========================

_MARKDOWN_FENCE_RE = re.compile(r"```+[\w\-]*", re.I)
_MARKDOWN_WORD = re.compile(r"\bmarkdown\b", re.I)
_NUM_RE = re.compile(r"\b-?\d+(?:[.,]\d+)?%?\b")

def _norm_line_for_compare(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = _MARKDOWN_FENCE_RE.sub("", s)
    s = re.sub(r"^[#\-\*\•\·\u2022\s]+", "", s)
    s = _MARKDOWN_WORD.sub("", s)
    s = s.replace(" & ", " and ")
    s = re.sub(r"[\.,:;()\[\]\"']", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _line_set_jaccard(lines_a: List[str], lines_b: List[str]) -> float:
    sa = { _norm_line_for_compare(l) for l in (lines_a or []) if _norm_line_for_compare(l) }
    sb = { _norm_line_for_compare(l) for l in (lines_b or []) if _norm_line_for_compare(l) }
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union

def _fuzzy_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _HAVE_RAPIDFUZZ:
        try:
            return float(fuzz.token_set_ratio(a, b) / 100.0)
        except Exception:
            pass
    return float(difflib.SequenceMatcher(None, a, b).ratio())

def _avg_line_similarity(lines_a: List[str], lines_b: List[str]) -> float:
    A = [ _norm_line_for_compare(x) for x in (lines_a or []) if _norm_line_for_compare(x) ]
    B = [ _norm_line_for_compare(x) for x in (lines_b or []) if _norm_line_for_compare(x) ]
    if not A and not B:
        return 0.0
    if not A or not B:
        return 0.0
    if len(A) <= len(B):
        small, large = A, B
    else:
        small, large = B, A
    sims = []
    for s in small:
        best = 0.0
        for t in large:
            val = _fuzzy_sim(s, t)
            if val > best:
                best = val
                if best >= 0.99:
                    break
        sims.append(best)
    if not sims:
        return 0.0
    return float(sum(sims) / len(sims))

def _numeric_similarity(text_a: str, text_b: str) -> float:
    if not text_a and not text_b:
        return 1.0
    nums_a = [n.replace(",", ".") for n in re.findall(_NUM_RE, text_a or "")]
    nums_b = [n.replace(",", ".") for n in re.findall(_NUM_RE, text_b or "")]
    if not nums_a and not nums_b:
        return 1.0
    if not nums_a or not nums_b:
        return 0.0
    matched_scores = []
    used = set()
    for na in nums_a:
        best_score = 0.0
        best_j = None
        try:
            va = float(na.rstrip("%"))
        except Exception:
            continue
        for j, nb in enumerate(nums_b):
            if j in used: continue
            try:
                vb = float(nb.rstrip("%"))
            except Exception:
                continue
            denom = max(abs(va), abs(vb), 1.0)
            rel = 1.0 - min(1.0, abs(va - vb) / denom)
            if rel > best_score:
                best_score = rel
                best_j = j
        if best_score > 0:
            used.add(best_j)
            matched_scores.append(best_score)
    if not matched_scores:
        return 0.0
    return float(sum(matched_scores) / len(matched_scores))

def _lineify_text(txt: str) -> List[str]:
    if not txt:
        return []
    s = txt.strip()
    s = _CODE_FENCE.sub(" ", s)
    s = _MARKDOWN_WORD.sub(" ", s)
    s = s.strip()
    if "\n" in s and s.count("\n") >= 1:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        return lines

    pairs = re.findall(r"([A-Za-z&/ \-]{2,60}?)\s*[:]?[\s]*([0-9]{1,4}%?)", s)
    if len(pairs) >= 2:
        out = []
        for lab, num in pairs:
            labn = lab.strip()
            if labn:
                out.append(f"{labn.strip()}: {num.strip()}")
        if out:
            return out

    if re.search(r"\s{2,}", s):
        parts = [p.strip() for p in re.split(r"\s{2,}", s) if p.strip()]
        if len(parts) >= 2:
            return parts

    if " - " in s or ";" in s:
        parts = [p.strip() for p in re.split(r" - |;", s) if p.strip()]
        if len(parts) >= 2:
            return parts

    if "|" in s:
        parts = [p.strip() for p in s.split("|") if p.strip()]
        if len(parts) >= 2:
            return parts

    return [s]

# =========================
# Image/chart helpers
# =========================

def _safe_open(path: str) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None

def _edge_density(pil_img: Image.Image) -> float:
    try:
        if not _HAVE_CV2:
            arr = np.asarray(pil_img.convert("L"))
            return float(np.var(arr) / (255.0**2))
        arr = np.asarray(pil_img.convert("L"))
        edges = cv2.Canny(arr, 50, 150)
        return float(edges.sum()) / (arr.shape[0]*arr.shape[1])
    except Exception:
        return 0.0

def _dominant_palette_hex(pil_img: Image.Image, k:int=4) -> List[str]:
    try:
        cols = dominant_colors(pil_img, k=k)
        def to_hex(c):
            if isinstance(c, tuple) or isinstance(c, list):
                r,g,b = c[:3]
                return "#{:02x}{:02x}{:02x}".format(int(r),int(g),int(b))
            return "#000000"
        return [to_hex(c) for c in cols]
    except Exception:
        return []

def _is_probable_chart(pil_img: Image.Image, ocr_text: str, edge_density: float, palette):
    ocr_tokens = len(re.findall(r"\w+", ocr_text or ""))
    num_palette = len(palette or [])
    if num_palette >= 3 and edge_density < 0.05 and ocr_tokens >= 1:
        return True
    if edge_density >= 0.02 and (num_palette >= 2 or ocr_tokens >= 2):
        return True
    if re.search(r"\b(legend|axis|chart|pie|bar|slice|percent|%)\b", ocr_text or "", re.I):
        return True
    return False

def _detect_img_type(pil_img: Image.Image, ocr_text: str) -> str:
    edge_density_val = _edge_density(pil_img)
    palette = _dominant_palette_hex(pil_img, k=5)
    ocr_tokens = len(re.findall(r"\w+", ocr_text or ""))
    is_pie = False
    try:
        if _HAVE_CV2:
            arr = np.asarray(pil_img.convert("L"))
            circles = cv2.HoughCircles(arr, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                       param1=50, param2=30, minRadius=10, maxRadius=min(arr.shape)//2)
            if circles is not None and len(circles) > 0 and len(palette) >= 3:
                is_pie = True
    except Exception:
        is_pie = False

    try:
        if _HAVE_CV2:
            arrc = np.asarray(pil_img.convert("L"))
            edges = cv2.Canny(arrc, 60, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=arrc.shape[1]//10, maxLineGap=15)
            if lines is not None:
                short_count = sum(1 for l in lines if abs(l[0][2]-l[0][0]) < arrc.shape[1]*0.6 and abs(l[0][3]-l[0][1]) < arrc.shape[0]*0.6)
                if short_count > 6:
                    return "flowchart"
    except Exception:
        pass

    if is_pie:
        return "pie"

    try:
        if _HAVE_CV2:
            arr = np.asarray(pil_img.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = 0
            H,W = gray.shape
            for c in contours:
                x,y,wc,hc = cv2.boundingRect(c)
                if wc > 0 and hc > 0 and hc > H*0.08 and wc < W*0.6:
                    rects += 1
            if rects >= 2 and edge_density_val > 0.01:
                return "bar"
    except Exception:
        pass

    if edge_density_val > 0.015 and len(palette) <= 4 and re.search(r"\b(x[- ]?axis|y[- ]?axis|axis|time|value|%|percent)\b", ocr_text or "", re.I):
        return "line"

    if re.search(r"\b\d{1,3}%\b", ocr_text or "") or ('|' in ocr_text):
        return "table_image"

    ents = re.findall(r"\b[A-Z][A-Z0-9]{1,6}\b", ocr_text or "")
    if len(ents) >= 2 and (edge_density_val > 0.01 or len(palette) <= 3):
        return "diagram"

    if _is_probable_chart(pil_img, ocr_text, edge_density_val, palette):
        return "bar" if len(palette) >= 3 else "line"

    return "photo" if edge_density_val < 0.01 and len(palette) > 10 else "unknown"

def _extract_legend_labels(pil_img: Image.Image):
    if not _HAVE_TESS:
        return "", []
    try:
        W,H = pil_img.size[0], pil_img.size[1]
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
        n = len(data['text'])
        boxes = []
        for i in range(n):
            txt = (data['text'][i] or "").strip()
            if not txt: continue
            x,y,w,h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            boxes.append((x,y,w,h,txt))
        candidates = [b for b in boxes if b[0] > 0.6*W or b[1] < 0.25*H]
        if not candidates:
            candidates = [b for b in boxes if b[0] > 0.5*W]
        labels = [b[4] for b in candidates]
        labels = [l for l in labels if len(re.findall(r"[A-Za-z0-9%]", l)) > 0]
        return " ".join(labels), labels
    except Exception:
        return "", []

def _extract_axes_text(pil_img: Image.Image):
    if not _HAVE_TESS or not _HAVE_CV2:
        return ""
    try:
        arr = np.asarray(pil_img.convert("L"))
        edges = cv2.Canny(arr, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=arr.shape[1]//4, maxLineGap=10)
        if lines is None:
            return ""
        H,W = arr.shape
        texts = []
        for l in lines:
            x1,y1,x2,y2 = l[0]
            x0 = max(0, min(x1,x2) - 10)
            x1b = min(W, max(x1,x2) + 10)
            y0 = max(0, min(y1,y2) - 20)
            y1b = min(H, max(y1,y2) + 20)
            band = pil_img.crop((x0,y0,x1b,y1b))
            t = pytesseract.image_to_string(band, config="--psm 6").strip()
            if t:
                texts.append(t)
        return " ".join(texts)
    except Exception:
        return ""

def _extract_chart_data_signature(pil_img: Image.Image, kind: str, bins: int = 16) -> np.ndarray:
    try:
        arr = np.asarray(pil_img.convert("RGB"))
        H,W,_ = arr.shape
        if kind == "pie":
            if _HAVE_CV2:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                           param1=50, param2=30, minRadius=10, maxRadius=min(H,W)//2)
                if circles is not None and len(circles) > 0:
                    cx,cy,r = map(int, circles[0][0][:3])
                    mask = np.zeros((H,W), dtype=np.uint8)
                    cv2.circle(mask, (cx,cy), r, 255, -1)
                    pts = arr[mask==255].reshape(-1,3)
                    if pts.shape[0] < 20:
                        return _tiny_color_hist(pil_img, bins=bins)
                    Z = np.float32(pts)
                    K = min(6, max(1, int(len(Z)/200)))
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
                    counts = np.bincount(labels.flatten(), minlength=K).astype(float)
                    vec = counts / counts.sum()
                    out = np.zeros(bins, dtype=float)
                    out[:len(vec)] = vec[:bins]
                    return out
            return _tiny_color_hist(pil_img, bins=bins)[:bins]
        elif kind == "bar":
            if _HAVE_CV2:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                _,th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                heights = []
                for c in contours:
                    x,y,wc,hc = cv2.boundingRect(c)
                    if hc > H*0.05 and wc < W*0.9:
                        heights.append(hc)
                if heights:
                    heights = np.array(sorted(heights)[-bins:], dtype=float)
                    heights = heights / (heights.max() or 1.0)
                    out = np.zeros(bins, dtype=float)
                    out[:len(heights)] = heights[:bins]
                    return out
            return _tiny_color_hist(pil_img, bins=bins)[:bins]
        elif kind == "line":
            if _HAVE_CV2:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                xs = np.linspace(0, W-1, num=bins, dtype=int)
                vals = []
                for x in xs:
                    col = edges[:, x]
                    vals.append(float(col.sum()) / (len(col) or 1))
                v = np.array(vals, dtype=float)
                if v.max() > 0:
                    v = v / v.max()
                out = np.zeros(bins, dtype=float)
                out[:len(v)] = v
                return out
            return _tiny_color_hist(pil_img, bins=bins)[:bins]
        else:
            return _tiny_color_hist(pil_img, bins=bins)[:bins]
    except Exception:
        return _tiny_color_hist(pil_img, bins=bins)[:bins]

# =========================
# Enrichment (robust: OCR + fallback + preserve newlines)
# =========================
###### OCR Async Azure OpenAI ######################

import asyncio
import concurrent.futures
from openai import AzureOpenAI
from typing import Any, List, Optional
import json
import base64
import io
import re
import os
from PIL import Image
import numpy as np

# (assume _safe_open, _normalize_ocr, _tiny_color_hist, dominant_colors,
#  _edge_density, _detect_img_type, _extract_legend_labels,
#  _extract_axes_text, _extract_chart_data_signature, _get_sbert_local,
#  _hash_vec are available in this module)

async def enrich_images_with_ocr_and_summary(
    artifacts: List[Any],
    image_root: str,
    debug: bool = False,
    side: Optional[str] = None,   # "v1" or "v2" to populate v1_text / v2_text
    max_side_px: int = 1200,      # (not used directly; _prepare_image controls sizing)
    concurrency: int = 8,         # max concurrent GPT calls
    task_id: Optional[str] = None,
    progress_start: int = 0,
    progress_end: int = 0,
    stage_name: str = "image enrichment",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """
    Async GPT-based image text extraction using AzureOpenAI (gpt-4o).
    Mutates artifacts in-place. Uses a semaphore to limit concurrent requests.
    """
    # Lazy import Azure client - keep same environment vars as before
    try:
        from openai import AzureOpenAI
    except Exception:
        AzureOpenAI = None

    AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", "")
    AZURE_KEY      = os.environ.get("AZURE_API_KEY", "")
    AZURE_VERSION  = os.environ.get("API_VERSION", "")
    DEPLOYMENT     = os.environ.get("AZURE_MODEL_GPT", "")

    azure_client = None
    if AzureOpenAI is not None and AZURE_ENDPOINT and AZURE_KEY and AZURE_VERSION and DEPLOYMENT:
        try:
            azure_client = AzureOpenAI(azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_KEY, api_version=AZURE_VERSION)
            ##logging.debug("[ENRICH][GPT] AzureOpenAI client initialized (async).")
        except Exception as e:
            azure_client = None
            logging.warning(f"[ENRICH][GPT] Failed to init Azure client: {e}. Falling back to artifact text.")
    else:
        logging.debug("[ENRICH][GPT] Azure config missing or openai package unavailable — using artifact fallback only.")

    # ---------------------------
    # PROMPT & HELPERS (from your working prompt)
    # ---------------------------

    SYSTEM_PROMPT = """
You are an OCR and diagram-reading expert.

Your job:
1. Read EXACTLY the text that appears inside the image and output EXACTLY as seen.
2. Include labels, axis titles, percentages, numbers, node names, box text, arrow labels, captions.
3. DO NOT hallucinate anything not present.
4. DO NOT describe the image.
5. DO NOT summarize inside the extracted text.
6. DO NOT add formatting noise:
   - no markdown
   - no bullets
   - no pipes
   - no headings
   - no code fences
   - no lists
7. Preserve reading order: if text appears in multiple regions, output them in top → bottom, left → right order, separating regions by a single blank line.
8. For diagrams/flowcharts: list each node text on its own line in reading order; if arrows or relationships are shown, include arrow relationships as plain inline text like: "Node A -> Node B".
9. For charts/graphs: include axis labels and then list data points as comma-separated pairs on separate lines (e.g. "1970, 814").
10. Keep capitalization EXACTLY as seen.
11. Output ONLY plain text with natural newlines.

If text is unreadable, return an empty string.
"""

    MAX_PAYLOAD_BYTES = 200_000
    START_MAX_SIDE = 1600
    MIN_MAX_SIDE = 600
    START_QUALITY = 95
    MIN_QUALITY = 70

    def _to_base64_jpeg(img: Image.Image, quality: int):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        b = buf.getvalue()
        return base64.b64encode(b).decode("ascii"), len(b)

    def _prepare_image(img: Image.Image):
        w, h = img.size
        max_side = min(START_MAX_SIDE, max(w, h))
        quality = START_QUALITY

        b64, size = _to_base64_jpeg(img, quality)
        if size <= MAX_PAYLOAD_BYTES:
            return b64

        while True:
            scale = max_side / float(max(w, h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img2 = img.resize((new_w, new_h), Image.LANCZOS)

            q = quality
            while q >= MIN_QUALITY:
                b64, size = _to_base64_jpeg(img2, q)
                if size <= MAX_PAYLOAD_BYTES:
                    return b64
                q = int(q * 0.8)

            max_side = int(max_side * 0.75)
            if max_side < MIN_MAX_SIDE:
                b64, _ = _to_base64_jpeg(img2, MIN_QUALITY)
                return b64

    def _strip_markdown_wrappers(s: str) -> str:
        if not s:
            return ""
        t = re.sub(r"```+", "", s)
        return t.strip()

    sbert = get_sbert_model()

    async def _publish_stream_update_async(message: str, completed_ratio: int):
        if progress_callback is not None and task_id:
            progress_callback({
                "type": "stream",
                "task_id": task_id,
                "kind": "image",
                "completed_ratio": completed_ratio,
                "message": message,
            })
            return
        if not task_id:
            return
        result = {
            "type": "stream",
            "task_id": task_id,
            "kind": "image",
            "completed_ratio": completed_ratio,
            "message": message,
        }
        queue_name = f"delta_comparator_{task_id}"
        await create_queue_publish(queue_name, result)

    # Helper to extract artifact fallback text
    def _artifact_fallback(a: Any) -> str:
        fb = ""
        for attr in ("text", "text_orig", "_provided_text", "provided_text", "raw_text", "_raw_text"):
            val = getattr(a, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        for attr in ("payload", "_payload", "meta", "_meta"):
            val = getattr(a, attr, None)
            if isinstance(val, dict):
                for k in ("text", "text_orig", "content"):
                    vv = val.get(k)
                    if isinstance(vv, str) and vv.strip():
                        return vv.strip()
        return ""

    # Prepare list of image artifacts to process (and quick fallbacks)
    image_items = []
    for a in artifacts:
        if getattr(a, "type", "") != "image":
            continue
        p = getattr(a, "img_path", "") or ""
        image_items.append((a, p))

    total_items = len(image_items)
    processed_items = 0
    next_checkpoint = 10
    progress_lock = asyncio.Lock()
    liveness_interval_seconds = 12

    async def _mark_item_processed():
        nonlocal processed_items, next_checkpoint
        if not total_items:
            return
        async with progress_lock:
            processed_items += 1
            percent = int(processed_items / total_items * 100)
            if percent < next_checkpoint and processed_items < total_items:
                return
            stage_ratio = progress_end if processed_items >= total_items else progress_start + int((progress_end - progress_start) * percent / 100)
            message = f"{stage_name}: {processed_items}/{total_items} images processed"
            while next_checkpoint <= percent:
                next_checkpoint += 10
        await _publish_stream_update_async(message, stage_ratio)

    async def _publish_liveness_updates():
        if not total_items:
            return
        try:
            while True:
                await asyncio.sleep(liveness_interval_seconds)
                async with progress_lock:
                    if processed_items >= total_items:
                        return
                    current_processed = processed_items
                    stage_ratio = progress_start + int((progress_end - progress_start) * current_processed / total_items)
                await _publish_stream_update_async(
                    f"{stage_name} still running: {current_processed}/{total_items} images processed",
                    stage_ratio,
                )
        except asyncio.CancelledError:
            return

    if total_items:
        await _publish_stream_update_async(f"{stage_name} started", progress_start)
    liveness_task = asyncio.create_task(_publish_liveness_updates()) if total_items else None

    # If no azure client, just do synchronous fallback processing
    if azure_client is None:
        #logging.debug("[ENRICH][GPT] No azure_client -> performing fallback-only enrichment.")
        try:
            for a, p in image_items:
                # open image best-effort but fallback if missing
                pil = None
                if p:
                    for c in (p, os.path.join(image_root, os.path.basename(p))):
                        pil = _safe_open(c)
                        if pil is not None:
                            break
                if pil is None:                    
                    fb = _artifact_fallback(a)
                    a.text_orig = fb or ""
                    a._img_text = _normalize_ocr(fb or "")
                    a._img_summary = ""
                    a._img_text_final = (a._img_text or "").strip()
                    if side == "v1": setattr(a, "v1_text", a.text_orig or "")
                    if side == "v2": setattr(a, "v2_text", a.text_orig or "")
                    #logging.debug(f"[ENRICH][FALLBACK] id={getattr(a,'id',None)} used artifact text.")
                    await _mark_item_processed()
                    continue

                a._img = pil
                a._img_arr = np.array(pil)
                fb = _artifact_fallback(a)
                a.text_orig = fb or ""
                a._img_text = _normalize_ocr(fb or "")
                a._img_summary = ""
                a._img_text_final = (a._img_text or "").strip()
                if side == "v1": setattr(a, "v1_text", a.text_orig or "")
                if side == "v2": setattr(a, "v2_text", a.text_orig or "")
                # compute chart features
                try:
                    a._img_color_hist = _tiny_color_hist(pil, bins=32)
                except Exception:
                    a._img_color_hist = np.zeros(96, dtype=float)
                try:
                    a._img_palette = dominant_colors(pil, k=6)
                except Exception:
                    a._img_palette = []
                a._edges_density = _edge_density(pil)
                a._img_type = _detect_img_type(pil, a._img_text)
                # chart fields (as before)
                a._chart_legend_text = ""
                a._chart_legend_labels = []
                a._chart_axis_text = ""
                a._chart_legend_emb = np.zeros(256, dtype=float)
                a._chart_axis_emb = np.zeros(256, dtype=float)
                a._chart_data_vector = np.zeros(16, dtype=float)
                a._chart_geometry = np.zeros(16, dtype=float)
                if a._img_type in ("pie", "bar", "line", "table_image"):
                    try:
                        labtxt, labs = _extract_legend_labels(pil)
                        a._chart_legend_text = labtxt
                        a._chart_legend_labels = labs
                    except Exception:
                        a._chart_legend_text = ""
                        a._chart_legend_labels = []
                    try:
                        a._chart_axis_text = _extract_axes_text(pil)
                    except Exception:
                        a._chart_axis_text = ""
                    try:
                        a._chart_data_vector = _extract_chart_data_signature(pil, a._img_type, bins=16)
                    except Exception:
                        a._chart_data_vector = np.zeros(16, dtype=float)
                    try:
                        a._chart_geometry = a._chart_data_vector.copy()
                    except Exception:
                        a._chart_geometry = np.zeros(16, dtype=float)
                    if sbert:
                        # try:
                                            
                        #     emb_legend = sbert.encode(" ".join(a._chart_legend_labels) or a._chart_legend_text or "", normalize_embeddings=True)
                                            
                        #     emb_axis = sbert.encode(a._chart_axis_text or "", normalize_embeddings=True)
                        #     a._chart_legend_emb = np.asarray(emb_legend, dtype=float)
                        #     a._chart_axis_emb = np.asarray(emb_axis, dtype=float)
                        # except Exception:
                        #     a._chart_legend_emb = _hash_vec(a._chart_legend_text or "", dim=256)
                        #     a._chart_axis_emb = _hash_vec(a._chart_axis_text or "", dim=256)
                        
                        text_legend = " ".join(a._chart_legend_labels) or a._chart_legend_text or ""
                        text_axis = a._chart_axis_text or ""

                        try:
                            emb_legend = sbert.encode(text_legend, normalize=True)
                            emb_axis = sbert.encode(text_axis, normalize=True)

                            # encode() returns shape (1, dim) for single string
                            a._chart_legend_emb = np.asarray(emb_legend[0], dtype=float)
                            a._chart_axis_emb = np.asarray(emb_axis[0], dtype=float)

                        except Exception:
                            a._chart_legend_emb = _hash_vec(text_legend, dim=256)                                                                         
                            a._chart_axis_emb = _hash_vec(text_axis, dim=256)

                    else:
                        a._chart_legend_emb = _hash_vec(a._chart_legend_text or "", dim=256)
                        a._chart_axis_emb = _hash_vec(a._chart_axis_text or "", dim=256)                                
            return

        finally:
            if liveness_task:
                liveness_task.cancel()
                with suppress(asyncio.CancelledError):
                    await liveness_task

    # ---------------------------
    # Async worker: call GPT in thread and write back result to artifact
    # ---------------------------
    semaphore = asyncio.Semaphore(concurrency)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)

    async def _process_one(a: Any, img_path: str):
        """
        Prepare image, call GPT via azure_client in a thread, parse result, write back to artifact a.
        """
        # open image
        pil = None
        if img_path:
            for c in (img_path, os.path.join(image_root, os.path.basename(img_path))):
                pil = _safe_open(c)
                if pil is not None:
                    break

        if pil is None:
            # fallback-only; write artifact fallback and return
            fb = _artifact_fallback(a)
            a.text_orig = fb or ""
            a._img_text = _normalize_ocr(fb or "")
            a._img_summary = ""
            a._img_text_final = (a._img_text or "").strip()
            if side == "v1": setattr(a, "v1_text", a.text_orig or "")
            if side == "v2": setattr(a, "v2_text", a.text_orig or "")
            #logging.debug(f"[ENRICH][OPEN_FAIL][FALLBACK] id={getattr(a,'id',None)} used artifact fallback (no image).")
            # compute features from nothing (keep defaults)
            await _mark_item_processed()
            return

        # keep small refs
        a._img = pil
        a._img_arr = np.array(pil)

        # compression/resizing and prepare data url
        try:
            b64 = _prepare_image(pil)
            data_url = f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            logging.warning(f"[ENRICH][PREP_FAIL] id={getattr(a,'id',None)} prepare_image failed: {e}")
            # fallback
            fb = _artifact_fallback(a)
            a.text_orig = fb or ""
            a._img_text = _normalize_ocr(fb or "")
            a._img_summary = ""
            a._img_text_final = (a._img_text or "").strip()
            if side == "v1": setattr(a, "v1_text", a.text_orig or "")
            if side == "v2": setattr(a, "v2_text", a.text_orig or "")
            await _mark_item_processed()
            return

        # build messages
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}}
            ]},
        ]

        # call azure client in a thread under semaphore
        raw_output = ""
        usage_info = None
        gpt_text = ""
        used_fallback = False

        async with semaphore:
            try:
                # call in thread to avoid blocking event loop
                def _call_sync():
                    # Use azure_client from outer scope (synchronous)
                    return azure_client.chat.completions.create(
                        model=DEPLOYMENT,
                        messages=msgs,
                        max_tokens=4096,
                        temperature=0.0,
                    )
                response = await asyncio.get_running_loop().run_in_executor(executor, _call_sync)
                # parse response
                raw_output = (response.choices[0].message.content or "").strip()
                raw_output = _strip_markdown_wrappers(raw_output) if 'raw_output' in locals() else re.sub(r"```+", "", raw_output)
                raw_text = raw_output.strip()
                if not raw_text:
                    raise RuntimeError("GPT returned empty extraction")
                gpt_text = raw_text
                usage_info = getattr(response, "usage", None)
                used_fallback = False
            except Exception as e:
                used_fallback = True
                logging.warning(f"[ENRICH][GPT][FAIL] id={getattr(a,'id',None)} err={repr(e)}; will use artifact fallback.")
                # fallthrough to artifact fallback

        # artifact fallback if needed
        if used_fallback:
            final_text = _artifact_fallback(a) or gpt_text or ""
        else:
            final_text = gpt_text

        # write artifact fields
        a.text_orig = final_text or ""
        a._img_text = _normalize_ocr(final_text or "")
        a._img_summary = ""                # no summary kept
        a._img_text_final = (a._img_text or "").strip()
        if side == "v1": setattr(a, "v1_text", a.text_orig or "")
        if side == "v2": setattr(a, "v2_text", a.text_orig or "")

        # debug logs
        if debug:
            srcid = getattr(a, "id", None)
            chosen = "gpt" if (not used_fallback and gpt_text) else "fallback"
            #logging.debug(f"[ENRICH][{chosen.upper()}] id={srcid} chosen_source={chosen}")
            #logging.debug("=== RAW EXTRACTED TEXT (first 1000 chars) ===")
            #logging.debug((final_text or "(empty)")[:1000])
            if usage_info:
                try:
                    json_dmp = json.dumps(usage_info, indent=2)
                    #logging.debug(f"=== TOKEN USAGE ===\n{json_dmp}")
                except Exception:
                    logging.debug(f"usage: {usage_info}")
            #logging.debug("--------------------------------------------------")

        # compute image/chart features (synchronous, cheap)
        try:
            a._img_color_hist = _tiny_color_hist(pil, bins=32)
        except Exception:
            a._img_color_hist = np.zeros(96, dtype=float)
        try:
            a._img_palette = dominant_colors(pil, k=6)
        except Exception:
            a._img_palette = []
        a._edges_density = _edge_density(pil)
        a._img_type = _detect_img_type(pil, a._img_text)

        # chart-specific fields (as before)
        a._chart_legend_text = ""
        a._chart_legend_labels = []
        a._chart_axis_text = ""
        a._chart_legend_emb = np.zeros(256, dtype=float)
        a._chart_axis_emb = np.zeros(256, dtype=float)
        a._chart_data_vector = np.zeros(16, dtype=float)
        a._chart_geometry = np.zeros(16, dtype=float)

        if a._img_type in ("pie", "bar", "line", "table_image"):
            try:
                labtxt, labs = _extract_legend_labels(pil)
                a._chart_legend_text = labtxt
                a._chart_legend_labels = labs
            except Exception:
                a._chart_legend_text = ""
                a._chart_legend_labels = []
            try:
                a._chart_axis_text = _extract_axes_text(pil)
            except Exception:
                a._chart_axis_text = ""
            try:
                a._chart_data_vector = _extract_chart_data_signature(pil, a._img_type, bins=16)
            except Exception:
                a._chart_data_vector = np.zeros(16, dtype=float)
            try:
                a._chart_geometry = a._chart_data_vector.copy()
            except Exception:
                a._chart_geometry = np.zeros(16, dtype=float)
            if sbert:
                text_legend = " ".join(a._chart_legend_labels) or a._chart_legend_text or ""
                text_axis = a._chart_axis_text or ""

                try:
                    emb_legend = sbert.encode(text_legend, normalize=True)
                    emb_axis = sbert.encode(text_axis, normalize=True)

                    # encode() returns shape (1, dim) for single string
                    a._chart_legend_emb = np.asarray(emb_legend[0], dtype=float)
                    a._chart_axis_emb = np.asarray(emb_axis[0], dtype=float)

                except Exception:
                    a._chart_legend_emb = _hash_vec(text_legend, dim=256)
                    a._chart_axis_emb = _hash_vec(text_axis, dim=256)
            else:
                a._chart_legend_emb = _hash_vec(a._chart_legend_text or "", dim=256)
                a._chart_axis_emb = _hash_vec(a._chart_axis_text or "", dim=256)

        await _mark_item_processed()

    # Schedule tasks
    tasks = [ _process_one(a, p) for (a, p) in image_items ]
    if tasks:
        try:
            await asyncio.gather(*tasks)
        finally:
            if liveness_task:
                liveness_task.cancel()
                with suppress(asyncio.CancelledError):
                    await liveness_task

    # shutdown executor
    executor.shutdown(wait=False)
    return

# =========================
# Matching (generalized & robust)
# =========================

@dataclass
class _Cand:
    i: int
    j: int
    sid: str
    rid: str
    score: float
    imgsim: float
    textsim: float
    entity_j: float
    line_j: float
    atom_j: float

def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    try:
        nu = float(np.linalg.norm(u)); nv = float(np.linalg.norm(v))
        if nu == 0.0 or nv == 0.0:
            return 0.0
        return float(np.dot(u, v) / (nu * nv))
    except Exception:
        return 0.0

def _filename_base(p: str) -> str:
    return os.path.splitext(os.path.basename(p or ""))[0].lower()

def match_images_prepass(
    A: List[Any],
    B: List[Any],
    #threshold: float = 0.20,
    threshold: float = 0.55,
    k: int = 24,
    debug: bool = False,
    global_img_weight: float = GLOBAL_IMG_WEIGHT,
    global_text_weight: float = GLOBAL_TEXT_WEIGHT,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Generalized image matcher:
      - union shortlist: top-k by image ∪ top-k by text ∪ fuzzy-line matches
      - chart-aware scoring using legend/data/color/geometry signals
      - configurable global weights (global_img_weight, global_text_weight)
    """
    Aimg = [a for a in A if getattr(a, "type", "") == "image"]
    Bimg = [b for b in B if getattr(b, "type", "") == "image"]

    # Load CLIP embedder (checks env var OPENCLIP_MODEL_DIR, falls back to repo-relative, or uses histogram fallback)
    device = "cpu"
    clip = get_image_embedder(device=device)  # or pass device="gpu"
    sbert = get_sbert_model()

    def _img_emb_list(items):
        out = []
        for it in items:
            im = getattr(it, "_img", None)
            if im is None:
                out.append(np.zeros(96, dtype=np.float32))
            else:
                try:
                    v = clip.embed(im)
                    v = np.asarray(v, dtype=np.float32)
                    if v.size == 0 or np.linalg.norm(v) == 0.0:
                        ch = getattr(it, "_img_color_hist", None)
                        if ch is not None and np.linalg.norm(ch) > 0:
                            out.append(np.asarray(ch, dtype=np.float32))
                        else:
                            out.append(np.zeros(96, dtype=np.float32))
                    else:
                        out.append(v)
                except Exception:
                    ch = getattr(it, "_img_color_hist", None)
                    if ch is not None:
                        out.append(np.asarray(ch, dtype=np.float32))
                    else:
                        out.append(np.zeros(96, dtype=np.float32))
        return out

    def _txt_emb_list(items, use_final: bool):
        texts = []
        for it in items:
            if use_final:
                texts.append((getattr(it, "_img_text_final", "") or "").strip())
            else:
                texts.append((getattr(it, "_img_text", "") or "").strip())
        if sbert is None:
            return [_hash_vec(t, 256) for t in texts]
        try:
            embs = sbert.encode(texts, normalize=True, show_progress_bar=False)
            #logging.debug("USING SBERT for IMAGES")
            return [np.asarray(e, dtype=np.float32) for e in embs]
        except Exception:
            return [_hash_vec(t, 256) for t in texts]

    A_imgE = _img_emb_list(Aimg)
    B_imgE = _img_emb_list(Bimg)
    A_txtE_final = _txt_emb_list(Aimg, True)
    B_txtE_final = _txt_emb_list(Bimg, True)
    A_txtE_ocr = _txt_emb_list(Aimg, False)
    B_txtE_ocr = _txt_emb_list(Bimg, False)

    ###v1 values
    # subtype_thresholds = {
    #     "diagram": 0.25, "flowchart": 0.25,
    #     "pie": 0.60, "bar": 0.60, "line": 0.60, "table_image": 0.60,
    #     "photo": 0.35, "unknown": 0.30,
    # }

    subtype_thresholds = {
        "diagram": 0.55, "flowchart": 0.55,
        "pie": 0.60, "bar": 0.60, "line": 0.60, "table_image": 0.60,
        "photo": 0.55, "unknown": 0.55,
    }

    weights = {
        "diagram": {"img":0.60,"text":0.40},
        "flowchart":{"img":0.60,"text":0.40},
        "pie": {"legend":0.35,"data":0.40,"color":0.15,"img":0.10},
        "bar": {"axis":0.25,"data":0.40,"legend":0.20,"img":0.15},
        "line": {"axis":0.30,"geometry":0.45,"img":0.15,"color":0.10},
        "table_image": {"img":0.3,"text":0.4,"data":0.3},
        "unknown": {"img":0.55,"text":0.45},
        "photo": {"img":0.8,"text":0.2},
    }

    candidates: List[_Cand] = []
    IMG_EPS = 1e-5

    def safe_cos(u, v):
        try:
            if u is None or v is None or getattr(u, "size", None) == 0 or getattr(v, "size", None) == 0:
                return 0.0
            nu = np.linalg.norm(u); nv = np.linalg.norm(v)
            if nu == 0.0 or nv == 0.0: return 0.0
            return float(np.dot(u, v) / (nu * nv))
        except Exception:
            return 0.0

    for i, aa in enumerate(Aimg):
        if debug and False:
            logging.debug(f"[MATCH] Processing source image {getattr(aa,'id',None)} ({i+1}/{len(Aimg)})")
        img_sims = [(safe_cos(A_imgE[i], B_imgE[j]), j) for j in range(len(Bimg))]
        img_sims.sort(key=lambda x: -x[0])
        txt_sims = [(safe_cos(A_txtE_final[i], B_txtE_final[j]), j) for j in range(len(Bimg))]
        txt_sims.sort(key=lambda x: -x[0])

        top_img = [j for _, j in img_sims[:max(1, k)]]
        top_txt = [j for _, j in txt_sims[:max(1, k)]]

        linesA = getattr(aa, "_img_lines", []) or []
        fuzzy_js = set()
        if linesA:
            normA = [ _norm_line_for_compare(x) for x in linesA if _norm_line_for_compare(x) ]
            for j in range(len(Bimg)):
                linesB = getattr(Bimg[j], "_img_lines", []) or []
                normB = [ _norm_line_for_compare(x) for x in linesB if _norm_line_for_compare(x) ]
                if not normB:
                    continue
                if set(normA) & set(normB):
                    fuzzy_js.add(j)
                else:
                    best = 0.0
                    for a_line in normA:
                        for b_line in normB:
                            r = difflib.SequenceMatcher(None, a_line, b_line).ratio()
                            if r > best: best = r
                            if best >= 0.86: break
                        if best >= 0.86: break
                    if best >= 0.86:
                        fuzzy_js.add(j)

        shortlist = list(dict.fromkeys(top_img + top_txt + list(fuzzy_js)))

        if not img_sims or img_sims[0][0] <= IMG_EPS:
            shortlist = list(dict.fromkeys(top_txt + shortlist))

        if debug and False:
            logging.debug(f"[SHORTLIST] src={getattr(aa,'id',None)} max_img={img_sims[0][0] if img_sims else 0.0:.4f} shortlist_len={len(shortlist)}")

        entsA = getattr(aa, "_img_entities", [])
        atomsA = getattr(aa, "_img_atoms", [])

        for j in shortlist:
            bb = Bimg[j]
            img_cos = safe_cos(A_imgE[i], B_imgE[j])

            tcos1 = safe_cos(A_txtE_final[i], B_txtE_final[j])
            tcos2 = safe_cos(A_txtE_ocr[i],   B_txtE_ocr[j])
            text_cos = max(tcos1, tcos2)

            entsB = getattr(bb, "_img_entities", [])
            atomsB = getattr(bb, "_img_atoms", [])
            ent_j = _jaccard(entsA, entsB)
            line_j = _jaccard(getattr(aa, "_img_lines", []), getattr(bb, "_img_lines", []))
            atom_j = _jaccard(atomsA, atomsB)

            line_set_j = _line_set_jaccard(getattr(aa, "_img_lines", []), getattr(bb, "_img_lines", []))
            line_avg_sim = _avg_line_similarity(getattr(aa, "_img_lines", []), getattr(bb, "_img_lines", []))
            num_sim = _numeric_similarity(getattr(aa, "_img_text", "") or "", getattr(bb, "_img_text", "") or "")

            legend_sim = axis_sim = data_sim = color_sim = geometry_sim = 0.0
            a_legend_emb = getattr(aa, "_chart_legend_emb", None)
            b_legend_emb = getattr(bb, "_chart_legend_emb", None)
            if a_legend_emb is not None and b_legend_emb is not None:
                legend_sim = _cosine(a_legend_emb, b_legend_emb)
            a_axis_emb = getattr(aa, "_chart_axis_emb", None)
            b_axis_emb = getattr(bb, "_chart_axis_emb", None)
            if a_axis_emb is not None and b_axis_emb is not None:
                axis_sim = _cosine(a_axis_emb, b_axis_emb)
            a_data = getattr(aa, "_chart_data_vector", None)
            b_data = getattr(bb, "_chart_data_vector", None)
            if a_data is not None and b_data is not None:
                la = len(a_data); lb = len(b_data)
                if la == lb:
                    data_sim = _cosine(np.asarray(a_data), np.asarray(b_data))
                else:
                    L = max(la, lb)
                    aa_pad = np.zeros(L); aa_pad[:la] = a_data
                    bb_pad = np.zeros(L); bb_pad[:lb] = b_data
                    data_sim = _cosine(aa_pad, bb_pad)
            a_col = getattr(aa, "_img_color_hist", None); b_col = getattr(bb, "_img_color_hist", None)
            if a_col is not None and b_col is not None:
                try:
                    color_sim = max(0.0, 1.0 - np.linalg.norm(np.asarray(a_col) - np.asarray(b_col)))
                except Exception:
                    color_sim = 0.0
            a_geom = getattr(aa, "_chart_geometry", None)
            b_geom = getattr(bb, "_chart_geometry", None)
            if a_geom is not None and b_geom is not None:
                geometry_sim = _cosine(np.asarray(a_geom), np.asarray(b_geom))

            subtype = getattr(aa, "_img_type", None) or getattr(bb, "_img_type", None) or "unknown"
            if subtype not in subtype_thresholds:
                subtype = "unknown"
            w = weights.get(subtype, weights["unknown"])

            many_lines = (len(getattr(aa, "_img_lines", [])) >= 3 and len(getattr(bb, "_img_lines", [])) >= 3)
            if many_lines:
                list_text_score = 0.5 * line_set_j + 0.35 * line_avg_sim + 0.15 * num_sim
                list_text_score = max(list_text_score, 0.6 * ( (line_set_j+line_avg_sim)/2 ) + 0.4 * line_avg_sim )
                text_aug = max(text_cos, list_text_score)
            else:
                text_aug = max(text_cos, 0.6 * text_cos + 0.25 * line_avg_sim + 0.15 * num_sim)

            if len(entsA) >= 5 and len(entsB) >= 5 and ent_j >= 0.50:
                text_aug = max(text_aug, 0.70)
            text_aug = float(min(0.99, max(0.0, text_aug)))

            visual_score = img_cos
            textual_score = text_aug

            if subtype == "pie":
                chart_text_score = 0.5 * legend_sim + 0.5 * data_sim
                final_score = global_img_weight * visual_score + global_text_weight * chart_text_score
            elif subtype == "bar":
                chart_text_score = 0.4 * data_sim + 0.25 * axis_sim + 0.2 * legend_sim
                final_score = global_img_weight * visual_score + global_text_weight * chart_text_score
            elif subtype == "line":
                chart_text_score = 0.5 * geometry_sim + 0.3 * axis_sim
                final_score = global_img_weight * visual_score + global_text_weight * chart_text_score
            elif subtype == "table_image":
                chart_text_score = 0.5 * textual_score + 0.5 * data_sim
                final_score = global_img_weight * visual_score + global_text_weight * chart_text_score
            else:
                per_img = w.get("img", global_img_weight)
                per_text = w.get("text", global_text_weight)
                per_norm = (per_img + per_text) or 1.0
                per_img_norm = per_img / per_norm
                per_text_norm = per_text / per_norm
                weighted_local = per_img_norm * visual_score + per_text_norm * textual_score
                final_score = (global_img_weight * visual_score + global_text_weight * textual_score) * 0.7 + weighted_local * 0.3

            if _filename_base(getattr(aa, "img_path", "")) == _filename_base(getattr(bb, "img_path", "")):
                final_score += 0.01
            if getattr(aa, "parent", None) == getattr(bb, "parent", None):
                final_score += 0.005

            final_score = float(max(0.0, min(1.0, final_score)))
            thr = subtype_thresholds.get(subtype, threshold)

            ###FIXATING on the Threshold######
            #if debug:
                #logging.info(f"[CAND] src={getattr(aa,'id',None)} tgt={getattr(bb,'id',None)} subtype={subtype} img={img_cos:.3f} text={text_aug:.3f} legend={legend_sim:.3f} data={data_sim:.3f} color={color_sim:.3f} final={final_score:.4f} thr={thr:.3f}")

            #logging.debug(f"Final Score: {final_score}")
            #logging.debug(f"Threshold {thr}")
            if final_score >= thr:
                candidates.append(_Cand(
                    i=i, j=j,
                    sid=getattr(aa, "id"), rid=getattr(bb, "id"),
                    score=final_score,
                    imgsim=float(max(-1.0, min(1.0, img_cos))),
                    textsim=float(text_aug),
                    entity_j=float(ent_j),
                    line_j=float(line_j),
                    atom_j=float(atom_j),
                ))

    # Greedy 1-1 selection
    candidates.sort(key=lambda r: -r.score)
    picked, usedA, usedB = [], set(), set()
    for r in candidates:
        if r.sid in usedA or r.rid in usedB:
            continue
        picked.append(r)
        usedA.add(r.sid)
        usedB.add(r.rid)

    # Build out_rows
    out_rows: List[Dict[str, Any]] = []
    A_by_id = {getattr(a, "id"): a for a in Aimg}
    B_by_id = {getattr(b, "id"): b for b in Bimg}
    for r in picked:
        aa = A_by_id.get(r.sid)
        bb = B_by_id.get(r.rid)

        legend_sim = axis_sim = data_sim = color_sim = geometry_sim = 0.0
        if hasattr(aa, "_chart_legend_emb") and hasattr(bb, "_chart_legend_emb"):
            legend_sim = _cosine(getattr(aa, "_chart_legend_emb", np.zeros(256)), getattr(bb, "_chart_legend_emb", np.zeros(256)))
        if hasattr(aa, "_chart_axis_emb") and hasattr(bb, "_chart_axis_emb"):
            axis_sim = _cosine(getattr(aa, "_chart_axis_emb", np.zeros(256)), getattr(bb, "_chart_axis_emb", np.zeros(256)))
        if hasattr(aa, "_chart_data_vector") and hasattr(bb, "_chart_data_vector"):
            a_data = np.asarray(getattr(aa, "_chart_data_vector", np.zeros(16)))
            b_data = np.asarray(getattr(bb, "_chart_data_vector", np.zeros(16)))
            data_sim = _cosine(a_data, b_data)
        a_col = getattr(aa, "_img_color_hist", None); b_col = getattr(bb, "_img_color_hist", None)
        if a_col is not None and b_col is not None:
            try:
                color_sim = max(0.0, 1.0 - np.linalg.norm(np.asarray(a_col) - np.asarray(b_col)))
            except Exception:
                color_sim = 0.0

        linesA = getattr(aa, "_img_lines", []) or []
        linesB = getattr(bb, "_img_lines", []) or []
        line_set_j = _line_set_jaccard(linesA, linesB)
        line_avg_sim = _avg_line_similarity(linesA, linesB)
        contain_AB = 0.0
        if linesA:
            matches = 0
            for la in linesA:
                la_n = _norm_line_for_compare(la)
                if not la_n: continue
                for lb in linesB:
                    if _fuzzy_sim(la_n, _norm_line_for_compare(lb)) >= 0.86:
                        matches += 1
                        break
            contain_AB = matches / len(linesA) if linesA else 0.0
        num_sim = _numeric_similarity(getattr(aa, "_img_text", "") or "", getattr(bb, "_img_text", "") or "")

        out_rows.append({
            "source_id": r.sid,
            "revision_id": r.rid,
            "score": round(r.score, 4),
            "signals": {
                "imgsim": round(r.imgsim, 4),
                "textsim": round(r.textsim, 4),
                "entity_j": round(r.entity_j, 4),
                "line_j": round(r.line_j, 4),
                "atom_j": round(r.atom_j, 4),
                "line_set_j": round(float(line_set_j), 4),
                "line_avg_sim": round(float(line_avg_sim), 4),
                "contain_AB": round(float(contain_AB), 4),
                "num_sim": round(float(num_sim), 4),
                "legend_sim": round(float(legend_sim), 4),
                "axis_sim": round(float(axis_sim), 4),
                "data_sim": round(float(data_sim), 4),
                "color_sim": round(float(color_sim), 4),
                "geometry_sim": round(float(geometry_sim), 4),
                "img_weight_global": round(float(global_img_weight), 3),
                "text_weight_global": round(float(global_text_weight), 3),
                "img_sim": round(r.imgsim, 4),
                "text_sim": round(r.textsim, 4),
            },
            "source_meta": {"page_idx": getattr(aa, "page_idx", None), "parent": getattr(aa, "parent", None), "order": getattr(aa, "order", None)} if aa else {},
            "revision_meta": {"page_idx": getattr(bb, "page_idx", None), "parent": getattr(bb, "parent", None), "order": getattr(bb, "order", None)} if bb else {},
            "source_text": getattr(aa, "_img_summary", "") if aa else "",
            "revision_text": getattr(bb, "_img_summary", "") if bb else "",
            "source_img_path": getattr(aa, "img_path", None) if aa else None,
            "revision_img_path": getattr(bb, "img_path", None) if bb else None,
            "source_table_body": None,
            "revision_table_body": None,
        })

    unmatched = {
        "source": [getattr(a, "id") for a in Aimg if getattr(a, "id") not in {r["source_id"] for r in out_rows}],
        "revision": [getattr(b, "id") for b in Bimg if getattr(b, "id") not in {r["revision_id"] for r in out_rows}],
    }
    return out_rows, unmatched

__all__ = ["enrich_images_with_ocr_and_summary", "match_images_prepass"]

