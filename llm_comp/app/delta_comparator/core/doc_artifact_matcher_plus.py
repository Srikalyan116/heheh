#doc_artifact_matcher_plus.py
from __future__ import annotations
import os, re, math, hashlib, collections, random, unicodedata, difflib, html, statistics
from typing import List, Dict, Any, Tuple, Optional, Set
from app.delta_comparator.utils.logger import log as logging
from app.delta_comparator.core.onnx_sbert_loader import get_sbert_model
# from onnx_sbert_loader import _get_sbert_model

__all__ = [
    "build_artifacts", "build_indices", "build_auto_stopwords",
    "score_table", "score_table_robust", "score_image", "score_text",
    "TextVectorizer", "_shortlist", "hungarian",
    "table_structure_signature"
]

# ---------- Optional deps ----------
try:
    import numpy as np
except Exception:
    np = None

HAVE_SKLEARN = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
    HAVE_SKLEARN = True
except Exception:
    pass

HAVE_NEIGHBORS = False
if HAVE_SKLEARN:
    try:
        from sklearn.neighbors import NearestNeighbors
        HAVE_NEIGHBORS = True
    except Exception:
        HAVE_NEIGHBORS = False

HAVE_FAISS = False
try:
    import faiss  # type: ignore[import-not-found]
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

HAVE_BS4 = False
try:
    from bs4 import BeautifulSoup
    HAVE_BS4 = True
except Exception:
    pass

# ---------- SBERT (optional; disabled via env) ----------
_SBERT_MODEL = None

def _is_sbert_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "modules.json"))

def _is_transformer_dir(path: str) -> bool:
    # minimal check for a raw HF transformer folder
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    has_weights = any(
        os.path.isfile(os.path.join(path, f))
        for f in ("pytorch_model.bin", "model.safetensors")
    )
    return has_weights

def _maybe_load_sbert():
    """
    Try to load SBERT from local disk first.
    Supports:
      - SentenceTransformers format (has modules.json)
      - Plain HF transformer format (we wrap with Pooling)
    Fallback: download 'sentence-transformers/all-MiniLM-L6-v2'.
    Respects DISABLE_SBERT=1.
    """
    global _SBERT_MODEL
    if _SBERT_MODEL is not None:
        return _SBERT_MODEL

    if os.environ.get("DISABLE_SBERT", "0") == "1":
        #logging.debug("[SBERT] Disabled via DISABLE_SBERT=1")
        return None

    try:
        from sentence_transformers import SentenceTransformer, models
    except Exception as e:
        logging.error(f"[SBERT] sentence-transformers not installed or broken: {e}")
        return None

    # Candidate local paths to try (edit to your layout as needed)
    here = os.path.dirname(__file__)
    parent = os.path.dirname(here)
    candidates = [
        # 1) Your earlier path (make sure this folder is the *model* folder, not its parent)
        # 2) Simpler layout: ./models/all-MiniLM-L6-v2
        os.path.join(parent, "models", "all-MiniLM-L6-v2"),
        # 3) Env override: SBERT_LOCAL_DIR can point directly at a model folder
    ]
    candidates = [p for p in candidates if p]  # drop None/empty
    #logging.debug(f"Candidate: {candidates}")
    # Try local candidates
    for path in candidates:
        if not os.path.isdir(path):
            continue

        # Case A: proper SentenceTransformers directory
        if _is_sbert_dir(path):
            try:
                #logging.debug(f"[SBERT] Loading SentenceTransformers model from: {path}")
                _SBERT_MODEL = SentenceTransformer(path)
                return _SBERT_MODEL
            except Exception as e:
                logging.error(f"[SBERT] Failed to load SBERT-format model at {path}: {e}")
                # continue to next candidate

        # Case B: raw HF transformer directory (wrap with Pooling)
        if _is_transformer_dir(path):
            try:
                #logging.debug(f"[SBERT] Found raw HF transformer at: {path} — wrapping with Pooling")
                word = models.Transformer(path)
                pooling = models.Pooling(word.get_word_embedding_dimension())
                _SBERT_MODEL = SentenceTransformer(modules=[word, pooling])
                return _SBERT_MODEL
            except Exception as e:
                logging.error(f"[SBERT] Failed to wrap HF transformer at {path}: {e}")
                # continue to next candidate

        # If we got here, the folder exists but is not a valid model folder
        #logging.debug(f"[SBERT] Not a recognized model folder: {path}")

    # Fallback: download from HF
    try:
        #logging.debug("[SBERT] No usable local model — downloading 'sentence-transformers/all-MiniLM-L6-v2' …")
        _SBERT_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return _SBERT_MODEL
    except Exception as e:
        logging.error(f"[SBERT] Fallback download failed: {e}")
        return None

# ---------- Normalization / tokenization ----------
MD_MARKERS = ("```markdown", "```", "**", "__", "*", "_", "~", "#")

MOJIBAKE_FIXES = {
    "â—": "•", "â€¢": "•",
    "âž”": "•",
    "Â": "",
    "â€“": "–", "â€”": "—",
    "â€˜": "'", "â€™": "'", "â€œ": '"', "â€": '"',
    "â€¦": "...",
    "â€“": "-", "â€”": "-",
    "â€": '"',
    "â€¢": "•",
    "Ã—": "×",
    "âˆ’": "-",
    "â„¢": "™",
    "â‚‚": "₂",
}

_SUP_TAG_RE = re.compile(r"<\s*sup\b[^>]*>(.*?)</\s*sup\s*>", flags=re.IGNORECASE | re.DOTALL)
_BR_TAG_RE = re.compile(r"<\s*br\s*/?\s*>", flags=re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+(\s*\{[^{}]*\})?")
_DOLLAR_MATH_RE = re.compile(r"\$[^$]*\$")
_BRACED_RE = re.compile(r"[{}]")
_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)*\b")
_SECTION_NUM_RE = re.compile(r"\b\d+(?:[./]\d+)*\b")

def _extract_did(txt):
    if not txt:
        return None

    txt = txt.upper()

    # robust patterns (handles OCR + spacing issues)
    patterns = [
        r'IO\s*DID\s*[:\-]?\s*([0-9A-F]{3,4})',
        r'DID\s*[:\-]?\s*([0-9A-F]{3,4})',
    ]

    for p in patterns:
        m = re.search(p, txt)
        if m:
            return m.group(1)

    return None

def _is_heading_like(tokens: List[str], raw_text: str) -> bool:
    if not tokens: return False
    if _HEADING_RE.match(raw_text or ""): return True
    if len(tokens) <= 6 and sum(tok.isalpha() for tok in tokens) >= max(1, len(tokens)-1): return True
    return False

def _strip_math_tex(s: str) -> str:
    s = _DOLLAR_MATH_RE.sub(" ", s)
    s = _LATEX_CMD_RE.sub(" ", s)
    s = _BRACED_RE.sub(" ", s)
    return s

def _normalize_units_symbols(sl: str) -> str:
    sl = re.sub(r"<\s*=", "<=", sl)
    sl = re.sub(r">\s*=", ">=", sl)
    sl = re.sub(r"\s*~\s*", " ", sl)
    sl = re.sub(r"\b(km)\s*/\s*(h)\b", r"\1/h", sl)
    sl = re.sub(r"\b(m)\s*/\s*(s)\b", r"\1/s", sl)
    return sl

def _join_spaced_digits(sl: str) -> str:
    sl = re.sub(r"(?<=\d)\s+(?=\d)", "", sl)
    sl = re.sub(r"\s*\\?%\b", "%", sl)
    return sl

def _htmlish_cleanup(s: str) -> str:
    if not s: return ""
    t = html.unescape(s)
    t = _SUP_TAG_RE.sub(r"\1", t)
    t = _BR_TAG_RE.sub(" ", t)
    t = _TAG_RE.sub(" ", t)
    return t

def _fix_mojibake(s: str) -> str:
    if not s: return ""
    for k, v in MOJIBAKE_FIXES.items():
        s = s.replace(k, v)
    return s

def _clean_for_match(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    s = _fix_mojibake(s)
    s = _htmlish_cleanup(s)
    s = s.replace("\u2013","-").replace("\u2014","-")
    s = s.replace("•", " ").replace("·", " ").replace("●", " ").replace("○", " ")
    s = s.replace("►", " ").replace("➤", " ").replace("➔", " ").replace("→", " ")
    s = s.replace("|", " ").replace("\t", " ")
    s = re.sub(r"[;•●○►➤➔→]+", " ", s)
    s = re.sub(r"\s*[\r\n]+\s*", " ", s)
    sl = s.lower()
    for m in MD_MARKERS: sl = sl.replace(m, " ")
    sl = sl.replace("`", " ")
    sl = re.sub(r"[-_]{2,}", " ", sl)
    sl = _strip_math_tex(sl)
    sl = _normalize_units_symbols(sl)
    sl = _join_spaced_digits(sl)
    sl = re.sub(r"(?<=[a-z])-+(?=[a-z])", " ", sl)
    sl = sl.replace("euro-ncap", "euro ncap")
    keep = set(":/+-.%<=>")
    sl = "".join(ch if (ch.isalnum() or ch.isspace() or ch in keep) else " " for ch in sl)
    return " ".join(sl.split())

def _norm_text(s: str) -> str:
    if s is None: return ""
    s = _clean_for_match(s)
    # normalize dotted/slashed numbers (section indices like 9.6, 9.6.1, 12/34)
    s = re.sub(_SECTION_NUM_RE, "<num>", s)
    return s

def _tokenize(s: str) -> List[str]:
    s = _norm_text(s)
    return [t for t in s.split() if t]

def _n_shingles(tokens: List[str], n: int = 4) -> List[str]:
    if len(tokens) < n: return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def _trigrams(tokens: List[str]) -> List[str]:
    return _n_shingles(tokens, 3)

def _char_similarity(a: str, b: str) -> float:
    if not a and not b: return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def _cosine_list(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a)!=len(b): return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na==0 or nb==0: return 0.0
    return dot/(na*nb)

def _dice(a: Set[str], b: Set[str]) -> float:
    if not a and not b: return 0.0
    return 2 * len(a & b) / max(1, (len(a) + len(b)))

def simhash(tokens: List[str], bits: int=64) -> int:
    if not tokens: return 0
    v = [0]*bits
    for t in tokens:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if ((h>>i)&1) else -1
    out = 0
    for i in range(bits):
        if v[i] >= 0: out |= (1<<i)
    return out

def norm_hamming(a: int, b: int, bits: int=64) -> float:
    return ((a ^ b).bit_count())/bits

# ---------- Numeric tokens ----------
_STD_PAT = re.compile(
    r"\b(?:GB|GB/T|QC/T|QC\-T|CNCA\-[A-Za-z0-9\-]+|MIIT|EIDC|CVTSC)"
    r"[^\s,;)]{0,20}?\d{2,4}(?:\-\d{2,4})?\b",
    flags=re.IGNORECASE
)

def _numeric_tokens(s: str) -> Set[str]:
    pats = [
        r"0x[0-9a-f]+",
        r"[0-9]+(?:/[0-9]+)+",
        r"[0-9]+(?:\.[0-9]+)+",
        r"[0-9a-f]{2,}",
    ]
    found: Set[str] = set()
    for p in pats:
        found |= set(re.findall(p, s, flags=re.I))
    found |= set(m.group(0).lower() for m in _STD_PAT.finditer(s))
    return found

# ---------- Auto stop-words ----------
def build_auto_stopwords(texts: List[str], df_cutoff: float = 0.6) -> Set[str]:
    cleaned = [_norm_text(t or "") for t in texts]
    n = max(1, len(cleaned))
    seen_per_doc = []
    for s in cleaned:
        toks = set(s.split())
        seen_per_doc.append(toks)
    df = collections.Counter()
    for toks in seen_per_doc:
        df.update(toks)
    stops: Set[str] = set()
    for tok, cnt in df.items():
        frac = cnt / n
        if frac > df_cutoff and not re.fullmatch(r"(0x[0-9a-f]+|[0-9a-f/\.]+)", tok):
            stops.add(tok)
    return stops

def filter_tokens(tokens: List[str], stopset: Set[str]) -> List[str]:
    if not stopset: return tokens
    return [t for t in tokens if t not in stopset]

# ---------- Value/Label split ----------
VAL_UNIT = {"byte","bytes","kb","mb","gb","ms","s","sec","hz","%"}

def _classify_token(tok: str) -> str:
    t = tok.lower()
    if re.fullmatch(r"0x[0-9a-f]+", t): return "VALUE"
    if re.fullmatch(r"[0-9]+(/[0-9]+)+", t): return "VALUE"
    if re.fullmatch(r"[0-9]+(\.[0-9]+)+", t): return "VALUE"
    if re.fullmatch(r"[0-9a-f]{2,}", t): return "VALUE"
    if t in VAL_UNIT or t in {"+","/","-"}: return "VALUE"
    return "LABEL"

def split_value_label_tokens(tokens: List[str]) -> tuple[List[str], List[str]]:
    vals, labs = [], []
    for tok in tokens:
        (vals if _classify_token(tok) == "VALUE" else labs).append(tok)
    return vals, labs

# ---------- Containment ----------
def _token_containment(a_tokens: List[str], b_tokens: List[str]) -> float:
    sa, sb = set(a_tokens), set(b_tokens)
    if not sa and not sb: return 0.0
    short = sa if len(sa) <= len(sb) else sb
    inter = sa & sb
    return len(inter) / max(1, len(short))

def _char_containment(a: str, b: str) -> float:
    if not a and not b: return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    sm = difflib.SequenceMatcher(None, shorter, longer).ratio()
    M_est = sm * (len(shorter) + len(longer)) / 2.0
    return min(1.0, M_est / max(1, len(shorter)))

# ---------- Table parsing / signature ----------
def _bigrams(tokens: List[str]) -> List[str]:
    if len(tokens) < 2: return []
    return [tokens[i] + " " + tokens[i+1] for i in range(len(tokens)-1)]

def _to_float_generic(s: str) -> Optional[float]:
    if not s: return None
    t = s.strip().lower()
    if t in {"na","n/a","--","-",""}: return None
    t = re.sub(r"[\*\#@]+$", "", t).strip()
    t = re.sub(r"(?<=\d)\s+(?=\d)", "", t)
    t = t.replace(",", "").replace(" ","").replace(" ","")
    m = re.match(r"^[+-]?\d+(?:\.\d+)?$", t)
    if not m: return None
    try:
        return float(t)
    except Exception:
        return None

def _collapse_multiline_table_cells(rows_cells: List[List[str]]) -> List[List[str]]:
    """
    Collapses continuation rows in markdown tables.
    A continuation row is identified by an empty first column.
    """
    collapsed = []
    last_idx = None

    for row in rows_cells:
        first = row[0].strip() if row and row[0] else ""
        if first:
            collapsed.append(row)
            last_idx = len(collapsed) - 1
        else:
            if last_idx is not None:
                for i in range(1, len(row)):
                    if row[i].strip():
                        collapsed[last_idx][i] += " " + row[i]
    return collapsed

def _parse_table(html_str: str) -> Dict[str, Any]:
    out = {
        "headers": [], "header_cols": [],
        "first_col": [], "flat_text": "",
        "col_concat": {}, "row_texts": [],
        "header_bigrams": [], "col_num_ratio": {},
        "grid_cells": [], "grid_nums": []
    }
    if not html_str: return out

    s = str(html_str)
    s = html.unescape(s)
    s = _BR_TAG_RE.sub("\n", s)

    def _finalize_from_rows(rows_cells: List[List[str]]):
        out["grid_cells"] = rows_cells
        maxc = max((len(r) for r in rows_cells), default=0)
        norm_rows = [r + [""]*(maxc-len(r)) for r in rows_cells]
        out["grid_cells"] = norm_rows

        cols_concat = collections.defaultdict(list)
        for r in norm_rows:
            for ci, val in enumerate(r):
                v = _norm_text(val)
                if v: cols_concat[ci].append(v)
        out["col_concat"] = {i: _norm_text(" ".join(vs)) for i, vs in cols_concat.items()}

        grid_nums = []
        for r in norm_rows:
            row_nums = [_to_float_generic(c) for c in r]
            grid_nums.append(row_nums)
        out["grid_nums"] = grid_nums

        if norm_rows:
            out["first_col"] = [ _norm_text(r[0]) for r in norm_rows if r and _norm_text(r[0]) ]

        out["row_texts"] = [" ".join(_norm_text(c) for c in r if _norm_text(c)) for r in norm_rows]

        col_ratio = {}
        for ci in range(maxc):
            col_vals = [ _norm_text(r[ci]) for r in norm_rows ]
            toks = _tokenize(" ".join(col_vals))
            if not toks:
                col_ratio[ci] = 0.0
            else:
                v = sum(1 for t in toks if _classify_token(t) == "VALUE")
                col_ratio[ci] = v / max(1, len(toks))
        out["col_num_ratio"] = col_ratio

    # HTML path
    if HAVE_BS4 and ("<table" in s.lower() or "</tr>" in s.lower()):
        try:
            soup = BeautifulSoup(s, "lxml")
            tbl = soup.find("table")
            if tbl:
                header_row = None
                for tr in tbl.find_all("tr"):
                    if tr.find("th"):
                        header_row = tr
                        break
                header_cols = []
                if header_row:
                    cells = header_row.find_all(["th","td"])
                    for c in cells:
                        header_cols.append(_norm_text(c.get_text(" ", strip=True)))
                out["header_cols"] = header_cols
                out["headers"] = _tokenize(" ".join(header_cols))
                out["header_bigrams"] = _bigrams(out["headers"])

                rows_cells = []
                # for tr in tbl.find_all("tr"):
                #     cells = tr.find_all(["td","th"])
                #     if not cells: continue
                #     row = [c.get_text(" ", strip=True) for c in cells]
                #     rows_cells.append(row)
                # _finalize_from_rows(rows_cells)
                for tr in tbl.find_all("tr"):
                    cells = tr.find_all(["td","th"])
                    if not cells: continue
                    row = [c.get_text(" ", strip=True) for c in cells]
                    rows_cells.append(row)

                rows_cells = _collapse_multiline_table_cells(rows_cells)
                _finalize_from_rows(rows_cells)

                out["flat_text"] = _norm_text(tbl.get_text(" ", strip=True))
                return out
        except Exception:
            pass

    # Markdown-ish
    text_only = _TAG_RE.sub(" ", s)
    lines = [ln.strip() for ln in text_only.splitlines() if ln.strip()]
    if any("|" in ln for ln in lines):
        header_line = next((ln for ln in lines if "|" in ln), "")
        header_cells = [c.strip() for c in re.sub(r"^\|?(.+?)\|?$", r"\1", header_line).split("|")]
        header_cols = [_norm_text(h) for h in header_cells]
        headers_norm = _tokenize(" ".join(header_cells))
        out["header_cols"] = header_cols
        out["headers"] = headers_norm
        out["header_bigrams"] = _bigrams(headers_norm)

        data_lines = [ln for ln in lines if "|" in ln and ln != header_line and not re.fullmatch(r"[:\-\s\|]+", ln)]
        rows_cells = []
        # for ln in data_lines:
        #     inner = re.sub(r"^\|?(.+?)\|?$", r"\1", ln)
        #     cells = [c.strip() for c in inner.split("|")]
        #     rows_cells.append(cells)
        # _finalize_from_rows(rows_cells)
        for ln in data_lines:
            inner = re.sub(r"^\|?(.+?)\|?$", r"\1", ln)
            cells = [c.strip() for c in inner.split("|")]
            rows_cells.append(cells)
        rows_cells = _collapse_multiline_table_cells(rows_cells)
        _finalize_from_rows(rows_cells)

        out["flat_text"] = _norm_text(text_only)
        return out

    out["flat_text"] = _norm_text(text_only)
    return out

def table_structure_signature(html: str) -> Dict[str,Any]:
    if not html:
        return {"rows":0,"cols":0,"tags":[],"header_tokens":set()}
    sig = {"rows":0,"cols":0,"tags":[],"header_tokens":set()}
    if HAVE_BS4:
        try:
            soup = BeautifulSoup(html, "lxml")
            rows = soup.find_all("tr"); sig["rows"]=len(rows)
            max_cols=0; tags=[]; headers=set()
            for r in rows:
                cells = r.find_all(["td","th"]); ccount=0
                for c in cells:
                    colspan=int(c.get("colspan","1") or 1); ccount += colspan
                    tags.append(c.name)
                    if c.name=="th":
                        headers |= set(_tokenize(c.get_text(" ", strip=True)))
                if ccount>max_cols: max_cols=ccount
            sig["cols"]=max_cols; sig["tags"]=tags; sig["header_tokens"]=headers
            return sig
        except Exception:
            pass
    rows = re.findall(r"<tr\b", html, flags=re.I)
    cols = re.findall(r"<t[dh]\b", html, flags=re.I)
    sig["rows"]=len(rows)
    sig["cols"]=max(1, len(cols)//max(1,len(rows)))
    sig["tags"]=["td"]*len(cols)
    sig["header_tokens"]=set(_tokenize(re.sub(r"<[^>]+>"," ", html))[:20])
    return sig

def teds_lite(sigA: Dict[str,Any], sigB: Dict[str,Any]) -> float:
    if not sigA or not sigB: return 0.0
    r = 1 - min(abs(sigA["rows"]-sigB["rows"])/max(1, max(sigA["rows"],sigB["rows"])), 1.0)
    c = 1 - min(abs(sigA["cols"]-sigB["cols"])/max(1, max(sigA["cols"],sigB["cols"])), 1.0)
    A = collections.Counter(sigA["tags"]); B = collections.Counter(sigB["tags"])
    inter = sum((A & B).values()); union = sum((A | B).values()) or 1
    tj = inter/union
    return max(0.0, min(1.0, 0.30*r + 0.30*c + 0.40*tj))

# ---------- Vectorizer ----------
class TextVectorizer:
    def __init__(self):
        ##self.sbert = _maybe_load_sbert()
        self.sbert = get_sbert_model()
        self.tfidf = None
        self.cache = {}

    def clear_cache(self):
        #print(f"[CACHE] Clearing cache. Size before: {len(self.cache)}")
        self.cache.clear()

    def fit_tfidf(self, corpus: List[str]):
        if not HAVE_SKLEARN: return
        try:
            n = len(corpus) if hasattr(corpus, "__len__") else 0
            max_df = 1.0 if n < 3 else 0.9
            self.tfidf = TfidfVectorizer(min_df=1, max_df=max_df, ngram_range=(1,2))
            self.tfidf.fit(corpus)
        except Exception:
            self.tfidf = None
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        key = ("embed", tuple(texts))
        if key in self.cache: return self.cache[key]
        if self.sbert:
            try:
                #from sentence_transformers import SentenceTransformer  # noqa
                #print("Using SBERT-ONNX Model for embediing")
                #print(f"[EMBED] Encoding {len(texts)} texts")
                #vecs = self.sbert.encode(texts, convert_to_numpy=bool(np is not None)).tolist()
                vecs = self.sbert.encode(texts)
                if hasattr(vecs, "tolist"):
                    vecs = vecs.tolist()

                self.cache[key] = vecs
                #print(f"[EMBED] Done Cache Size: {len(self.cache)}")
                return vecs
            except Exception:
                pass
        if self.tfidf is None:
            self.fit_tfidf(texts)
        if HAVE_SKLEARN and self.tfidf is not None:
            try:
                mat = self.tfidf.transform(texts)
                vecs = mat.toarray().tolist()
                self.cache[key] = vecs
                return vecs
            except Exception:
                pass
        voc = {}
        for s in texts:
            for t in _tokenize(s): voc[t] = voc.get(t,0) + 1
        keys = sorted(voc.keys())
        out = []
        for s in texts:
            c = collections.Counter(_tokenize(s))
            out.append([c.get(k,0) for k in keys])
        self.cache[key] = out
        return out

    def cosine(self, A: List[str], B: List[str]) -> List[List[float]]:
        a = self.embed(A); b = self.embed(B)
        if HAVE_SKLEARN and np is not None:
            try:
                return sk_cosine_similarity(np.array(a), np.array(b)).tolist()
            except Exception:
                pass
        return [[_cosine_list(x,y) for y in b] for x in a]

# ---------- Corpus stats ----------
def _idf_from_texts(texts: List[str], smooth: float = 1.0) -> Dict[str, float]:
    docs = [set(_tokenize(t or "")) for t in texts]
    df = collections.Counter()
    for d in docs: df.update(d)
    N = max(1, len(docs))
    idf = {tok: math.log((N + smooth) / (cnt + smooth)) + 1.0 for tok, cnt in df.items()}
    return idf

def _top_anchor_tokens(tokens: List[str], idf_map: Dict[str,float], k: int = 8) -> List[str]:
    if not tokens: return []
    uniq = list(set(tokens))
    uniq.sort(key=lambda t: -idf_map.get(t, 1.0))
    return uniq[:k]

class Artifact:
    def __init__(self, id, type, page_idx, parent, text, source_file,
                 img_path=None, html=None, stopset: Optional[Set[str]] = None,
                 idf_map: Optional[Dict[str,float]] = None,
                 order: int = 0):

        self.id = id
        self.type = type
        self.page_idx = int(page_idx) if page_idx is not None else -1
        self.parent = _norm_text(parent)
        self.text_orig = text or ""
        self.text = _norm_text(text or "")
        self.order = int(order)

        t_raw = (self.text_orig or "").strip()
        tok_cnt = len(self.text.split())
        starts_num = bool(re.match(r"^\s*[\d.]+", t_raw))
        ends_colon = t_raw.endswith(":")
        cap_ratio = 0.0
        words = re.findall(r"[A-Za-z]+", t_raw)
        if words:
            caps = sum(1 for w in words if w and w[0].isupper())
            cap_ratio = caps / max(1, len(words))
        self.is_heading = (tok_cnt <= 12) and (starts_num or ends_colon or cap_ratio >= 0.60)

        toks = _tokenize(self.text)
        toks = filter_tokens(toks, stopset or set())
        self.tokens = toks
        self.trigrams = _trigrams(self.tokens) if len(self.tokens)>=3 else []
        self.simhash_text = simhash(self.tokens)
        self.text_clean = " ".join(self.tokens)
        self.is_heading = _is_heading_like(self.tokens, self.text_orig or self.text)

        self.text_semantic = re.sub(_SECTION_NUM_RE, "<num>", self.text_clean)
        self.tokens_value, self.tokens_label = split_value_label_tokens(self.tokens)
        self.text_value = " ".join(self.tokens_value)
        self.text_label = " ".join(self.tokens_label)

        self.shingles4 = _n_shingles(self.tokens, 4)
        self.anchor_tokens = _top_anchor_tokens(self.tokens, idf_map or {}, k=8)

        self.source_file = _norm_text(source_file or "")
        self.img_path = None
        if isinstance(img_path, str) and img_path.strip():
            basename = os.path.basename(img_path).lstrip("./\\")
            self.img_path = os.path.join("images", basename).replace("\\", "/")

        self.html = html
        self.num_tokens = _numeric_tokens(self.text_clean)

        # ---- TABLE BRANCH ----
        if self.type == "table":
            sig = table_structure_signature(self.html or "")
            self.table_sig = sig
            self.header_tokens = set(sig.get("header_tokens", set()))

            parsed = _parse_table(self.html or self.text_orig or "")
            self.table_headers   = parsed["headers"]
            self.header_cols     = parsed["header_cols"]
            self.table_first_col = parsed["first_col"]
            self.table_flat_text = parsed["flat_text"] or self.text_clean
            self.col_concat      = parsed["col_concat"]
            self.row_texts       = parsed["row_texts"]
            self.header_bigrams  = parsed["header_bigrams"]
            self.col_num_ratio   = parsed["col_num_ratio"]
            self.grid_cells      = parsed["grid_cells"]
            self.grid_nums       = parsed["grid_nums"]

            self.row_sigs = []
            for r in self.row_texts:
                toks = _tokenize(r)
                key = " ".join(toks[:6]) if toks else ""
                if key:
                    self.row_sigs.append(key)

            cnt = collections.Counter(_tokenize(self.table_flat_text))
            self.table_terms = [t for t, _ in cnt.most_common(32)
                                if len(t) >= 3 and t not in set(self.parent.split())]

            self.num_fingerprint = _build_numeric_fingerprint(self.grid_nums)

            # ---- NEW CANONICAL SEMANTIC TABLE BODY ----
            parts = []

            # A) headers first (if present)
            if self.table_headers:
                h = " ".join(self.table_headers)
                parts.append(re.sub(r"\s+", " ", h).strip())

            # B) rows flattened & whitespace normalized
            for r in self.row_texts:
                r = re.sub(r"\s+", " ", r).strip()
                if r:
                    parts.append(r)

            # C) fallback to flat text
            if not parts and self.table_flat_text:
                body = re.sub(r"\s+", " ", self.table_flat_text).strip()
                parts.append(body)

            # final canonical body (lowercase)
            # self.table_semantic = " ".join(parts).lower()
            # 24_03_2026 emphasize headers + first column (structure!)
            parts_weighted = []

            # headers twice (important)
            if self.table_headers:
                parts_weighted.append(" ".join(self.table_headers))
                parts_weighted.append(" ".join(self.table_headers))

            # first column (VERY important)
            if self.table_first_col:
                parts_weighted.append(" ".join(self.table_first_col))

            # rest rows
            parts_weighted.extend(parts)

            # self.table_semantic = " ".join(parts_weighted).lower() 
            # # Add structural hints to help BERT
            structure_tokens = []

            if self.table_headers:
                structure_tokens += ["header"] * len(self.table_headers)

            if self.table_first_col:
                structure_tokens += ["rowlabel"] * len(self.table_first_col)

            self.table_semantic = (
                " ".join(parts_weighted) + " " + " ".join(structure_tokens)
            ).lower()           

        else:
            # ---- NON-TABLE ----
            self.table_sig = None
            self.header_tokens = set()
            self.table_headers = []
            self.header_cols = []
            self.table_first_col = []
            self.table_flat_text = self.text_clean
            self.col_concat = {}
            self.row_texts = []
            self.header_bigrams = []
            self.row_sigs = []
            self.table_terms = []
            self.col_num_ratio = {}
            self.grid_cells = []
            self.grid_nums = []
            self.num_fingerprint = {}
            self.table_semantic = None      

    @property
    def section_path(self):
        return f"{self.source_file} / {self.parent}"

# ---------- Numeric fingerprint helpers ----------
def _col_series(grid_nums: List[List[Optional[float]]]) -> Dict[int, List[Optional[float]]]:
    cols = collections.defaultdict(list)
    if not grid_nums: return {}
    maxc = max((len(r) for r in grid_nums), default=0)
    for r in grid_nums:
        row = r + [None]*(maxc-len(r))
        for ci, v in enumerate(row):
            cols[ci].append(v)
    return cols

def _quantiles_safe(xs: List[float]) -> List[float]:
    if not xs: return [0.0]*5
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    def pick(p):
        if n == 1: return xs_sorted[0]
        idx = min(n-1, max(0, int(round(p*(n-1)))))
        return xs_sorted[idx]
    q = [pick(p) for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
    med = q[2]
    iqr = max(1e-9, q[3]-q[1])
    return [(v - med)/iqr for v in q]

def _missing_rate(xs: List[Optional[float]]) -> float:
    n = len(xs) or 1
    miss = sum(1 for v in xs if v is None)
    return miss / n

def _is_index_like(xs: List[Optional[float]]) -> bool:
    vals = [v for v in xs if v is not None]
    if len(vals) < 3: return False
    ints = all(abs(v - round(v)) < 1e-9 for v in vals)
    mono = all(vals[i] < vals[i+1] for i in range(len(vals)-1))
    return ints and mono

def _spearman_sign(xs: List[Optional[float]], idx: List[Optional[float]]) -> float:
    xv = [x for x,i in zip(xs, idx) if x is not None and i is not None]
    iv = [i for x,i in zip(xs, idx) if x is not None and i is not None]
    if len(xv) < 3: return 0.0
    inc = sum(1 for i in range(len(xv)-1) if xv[i+1] > xv[i])
    dec = sum(1 for i in range(len(xv)-1) if xv[i+1] < xv[i])
    if inc+dec == 0: return 0.0
    return (inc - dec) / max(1, (inc + dec))

def _corr_matrix(cols: Dict[int,List[Optional[float]]]) -> List[List[float]]:
    keys = sorted(cols.keys())
    if not keys: return []
    mat = [[0.0]*len(keys) for _ in keys]
    for ai,i in enumerate(keys):
        for aj,j in enumerate(keys):
            if aj < ai:
                mat[ai][aj] = mat[aj][ai]
                continue
            yi, xj = [], []
            for vx, vy in zip(cols[i], cols[j]):
                if vx is not None and vy is not None:
                    yi.append(vx); xj.append(vy)
            if len(yi) >= 4:
                try:
                    mu1 = statistics.fmean(yi); mu2 = statistics.fmean(xj)
                    s1 = statistics.pstdev(yi) or 1e-9
                    s2 = statistics.pstdev(xj) or 1e-9
                    cov = statistics.fmean([(a-mu1)*(b-mu2) for a,b in zip(yi,xj)])
                    r = cov/(s1*s2)
                except Exception:
                    r = 0.0
            else:
                r = 0.0
            mat[ai][aj] = r; mat[aj][ai] = r
    return mat

def _frobenius_sim(A: List[List[float]], B: List[List[float]]) -> float:
    if not A or not B: return 0.0
    n = min(len(A), len(B)); m = min(len(A[0]), len(B[0]))
    if n==0 or m==0: return 0.0
    s = 0.0
    for i in range(n):
        for j in range(m):
            s += (A[i][j]-B[i][j])**2
    return 1.0 / (1.0 + s)

def _build_numeric_fingerprint(grid_nums: List[List[Optional[float]]]) -> Dict[str, Any]:
    cols = _col_series(grid_nums)
    if not cols:
        return {
            "col_quant": {}, "col_scale": {}, "col_missing_rate": {},
            "has_index": False, "index_col": -1, "trend_sign": {}, "corr": []
        }
    col_quant = {}
    col_scale = {}
    col_missing = {}
    for ci, xs in cols.items():
        vals = [v for v in xs if v is not None]
        col_missing[ci] = _missing_rate(xs)
        if vals:
            col_quant[ci] = _quantiles_safe(vals)
            try:
                medabs = statistics.fmedian([abs(v) for v in vals])
            except AttributeError:
                abs_vals = [abs(v) for v in vals]
                medabs = float(np.median(abs_vals)) if abs_vals else 0.0
            col_scale[ci] = math.log10(max(1e-9, medabs))
        else:
            col_quant[ci] = [0.0]*5
            col_scale[ci] = -9.0

    index_col = -1
    for ci, xs in cols.items():
        if _is_index_like(xs):
            index_col = ci; break
    trend_sign = {}
    if index_col >= 0:
        idx = cols[index_col]
        for ci, xs in cols.items():
            if ci == index_col: continue
            trend_sign[ci] = _spearman_sign(xs, idx)

    numeric_cols = {ci: xs for ci, xs in cols.items() if sum(v is not None for v in xs) >= 4}
    corr = _corr_matrix(numeric_cols) if len(numeric_cols) >= 2 else []

    return {
        "col_quant": col_quant, "col_scale": col_scale,
        "col_missing_rate": col_missing,
        "has_index": index_col >= 0, "index_col": index_col,
        "trend_sign": trend_sign, "corr": corr
    }

def _jaccard_stringset(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not (A or B): return 0.0
    return len(A & B)/max(1, len(A | B))

# ---------- Indices & shortlist ----------
def _semantic_text_for_kind(a: "Artifact", kind: str) -> str:
    if kind == "table":
        return getattr(a, "table_semantic", None) or getattr(a, "table_flat_text", "") or getattr(a, "text_clean", "")
    if kind == "text":
        return getattr(a, "text_semantic", None) or getattr(a, "text_clean", "")
    return getattr(a, "text_clean", "")

def _build_ann_index_for_kind(arts: List["Artifact"], kind: str, tv: Optional["TextVectorizer"]):
    if tv is None or np is None:
        return None

    ids = [i for i, a in enumerate(arts) if a.type == kind]
    if not ids:
        return None

    texts = [_semantic_text_for_kind(arts[i], kind) for i in ids]
    vecs = tv.embed(texts)
    if not vecs:
        return None

    mat = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    mat = mat / norms

    if HAVE_FAISS:
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        return {"backend": "faiss", "index": index, "ids": ids, "mat": mat}

    if HAVE_NEIGHBORS:
        nn = NearestNeighbors(metric="cosine", algorithm="auto")
        nn.fit(mat)
        return {"backend": "sklearn", "index": nn, "ids": ids, "mat": mat}

    return {"backend": "numpy", "index": None, "ids": ids, "mat": mat}

def _ann_topk_ids(ann_index: Dict[str, Any], qvec: List[float], top_k: int) -> List[int]:
    if not ann_index or top_k <= 0:
        return []

    ids = ann_index.get("ids", [])
    if not ids:
        return []

    k = min(top_k, len(ids))
    q = np.asarray(qvec, dtype=np.float32).reshape(1, -1)
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    np.maximum(qn, 1e-12, out=qn)
    q = q / qn

    backend = ann_index.get("backend")
    if backend == "faiss":
        _d, i = ann_index["index"].search(q, k)
        return [ids[x] for x in i[0] if x >= 0]

    if backend == "sklearn":
        _dist, i = ann_index["index"].kneighbors(q, n_neighbors=k)
        return [ids[x] for x in i[0] if x >= 0]

    mat = ann_index.get("mat")
    if mat is None or len(mat) == 0:
        return []
    sims = np.dot(mat, q[0])
    if k >= len(sims):
        ord_idx = np.argsort(-sims)
    else:
        top_idx = np.argpartition(-sims, k - 1)[:k]
        ord_idx = top_idx[np.argsort(-sims[top_idx])]
    return [ids[int(x)] for x in ord_idx]

def build_indices(arts: List["Artifact"], tv: Optional["TextVectorizer"] = None, kind: Optional[str] = None):
    idx_tri  = collections.defaultdict(set)
    idx_sh   = collections.defaultdict(set)
    idx_tok  = collections.defaultdict(set)
    idx_head = collections.defaultdict(set)
    idx_tail = collections.defaultdict(set)
    idx_hdr  = collections.defaultdict(set)
    idx_rsig = collections.defaultdict(set)
    idx_htbg = collections.defaultdict(set)
    idx_ttop = collections.defaultdict(set)
    idx_text = collections.defaultdict(set)
    idx_numtok = collections.defaultdict(set)
    kind_ids = collections.defaultdict(list)

    for i, a in enumerate(arts):
        kind_ids[a.type].append(i)
        for tri in a.trigrams: idx_tri[tri].add(i)
        idx_sh[(a.simhash_text >> 48) & 0xFFFF].add(i)
        for t in a.tokens: idx_tok[t].add(i)
        if a.tokens:
            idx_head[" ".join(a.tokens[:5])].add(i)
            idx_tail[" ".join(a.tokens[-5:])].add(i)

        # exact normalized text mapping (helps identical long sentences)
        if getattr(a, "text_clean", ""):
            idx_text[a.text_clean].add(i)

        if getattr(a, "num_tokens", None):
            for tok in a.num_tokens:
                idx_numtok[tok].add(i)

        if a.type == "table":
            for tok in a.table_headers:
                for t in _tokenize(tok):
                    if t: idx_hdr[t].add(i)
            for t in (a.header_tokens or set()):
                if t: idx_hdr[t].add(i)
            for bg in a.header_bigrams:
                idx_htbg[bg].add(i)
            for sig in a.row_sigs:
                for t in _tokenize(sig):
                    if t: idx_rsig[t].add(i)
            for t in a.table_terms[:24]:
                idx_ttop[t].add(i)

    out = {
        "tri": idx_tri, "sh": idx_sh, "tok": idx_tok,
        "head": idx_head, "tail": idx_tail,
        "hdr": idx_hdr, "rsig": idx_rsig, "htbg": idx_htbg, "ttop": idx_ttop,
        "text_exact": idx_text,
        "numtok": idx_numtok,
        "kind_ids": dict(kind_ids),
    }
    if kind:
        out["ann"] = _build_ann_index_for_kind(arts, kind, tv)
    return out

def _shortlist(a: "Artifact", B: List["Artifact"], indexB, kind: str, k: int = 16, tv: Optional["TextVectorizer"] = None, ann_top_k: int = 48) -> List[int]:
    cand = set()

    # immediate exact normalized-text matches (very important for identical sentences)
    try:
        exact_set = indexB.get("text_exact", {}).get(a.text_clean, set())
        cand |= set(exact_set)
    except Exception:
        pass

    ann = indexB.get("ann")
    if ann is not None and tv is not None and np is not None:
        qtxt = _semantic_text_for_kind(a, kind)
        qvec = tv.embed([qtxt])[0]
        cand |= set(_ann_topk_ids(ann, qvec, ann_top_k))

    for tri in a.trigrams:
        cand |= indexB["tri"].get(tri, set())
    cand |= indexB["sh"].get((a.simhash_text >> 48) & 0xFFFF, set())

    if kind == "text":
        base_tokens = a.anchor_tokens if len(a.tokens) <= 6 else a.tokens
        pulls = 12 if len(a.tokens) <= 6 else 24
        for t in base_tokens:
            if not t: continue
            cand |= set(list(indexB["tok"].get(t, set()))[:pulls])
        if a.tokens:
            cand |= indexB["head"].get(" ".join(a.tokens[:5]), set())
            cand |= indexB["tail"].get(" ".join(a.tokens[-5:]), set())

    elif kind == "table":
        hdr_toks = set(a.header_tokens)
        for ht in a.table_headers[:10]:
            hdr_toks |= set(_tokenize(ht))
        for tok in list(hdr_toks)[:24]:
            cand |= indexB["hdr"].get(tok, set())
        for bg in a.header_bigrams[:24]:
            cand |= indexB["htbg"].get(bg, set())
        for sig in a.row_sigs[:16]:
            for tok in _tokenize(sig):
                cand |= indexB["rsig"].get(tok, set())
        for t in a.table_terms[:16]:
            cand |= indexB["ttop"].get(t, set())
        if a.num_tokens:
            for tok in a.num_tokens:
                cand |= indexB["numtok"].get(tok, set())

    cands = [i for i in cand if B[i].type == kind]
    same_path = [i for i in cands if B[i].section_path == a.section_path]
    same_head = [i for i in cands if getattr(B[i], "is_heading", False) == getattr(a, "is_heading", False) and i not in same_path]
    others    = [i for i in cands if i not in same_path and i not in same_head]
    out = same_path + same_head + others

    # Keep candidate set bounded so score collection remains sub-quadratic.
    want_k = 64 if kind == "table" else (72 if kind == "text" else k)
    if len(out) < want_k:
        extras = indexB.get("kind_ids", {}).get(kind, [])
        if extras:
            missing = [x for x in extras if x not in out]
            missing.sort(key=lambda j: (
                int(B[j].parent != a.parent),
                abs(B[j].page_idx - a.page_idx),
                abs(getattr(B[j], "order", 0) - getattr(a, "order", 0)),
            ))
            out += missing[: (want_k - len(out))]

    return list(dict.fromkeys(out))[:want_k]

# ---------- Scorers ----------
def score_text(a, b, tv):
    # exact & same type → fast-path
    if getattr(a, "text_clean", "") == getattr(b, "text_clean", "") and getattr(a, "type", None) == getattr(b, "type", None):
        return 1.0, {
            "semantic": 1.0, "chars": 1.0, "contain_tok": 1.0, "contain_char": 1.0,
            "anchor_cover": 1.0, "anchor_jacc": 1.0, "shingle_cover": 1.0,
            "sem_labels": 1.0, "contain_labels": 1.0
        }

    # SAFE fallbacks for missing attributes
    a_text_clean = getattr(a, "text_clean", "")
    b_text_clean = getattr(b, "text_clean", "")
    a_text_sem   = getattr(a, "text_semantic", a_text_clean)
    b_text_sem   = getattr(b, "text_semantic", b_text_clean)
    a_text_lbl   = getattr(a, "text_label", None) or a_text_sem
    b_text_lbl   = getattr(b, "text_label", None) or b_text_sem

    a_tokens = getattr(a, "tokens", []) or []
    b_tokens = getattr(b, "tokens", []) or []

    # derive label-only tokens if absent
    a_tokens_label = getattr(a, "tokens_label", None)
    b_tokens_label = getattr(b, "tokens_label", None)
    if a_tokens_label is None or b_tokens_label is None:
        def _is_value(tok: str) -> bool:
            return bool(re.fullmatch(r"(0x[0-9a-f]+|[0-9]+(/[0-9]+)+|[0-9]+(\.[0-9]+)+|[0-9a-f]{2,}|%|kb|mb|gb|ms|s|sec|hz)", tok.lower()))
        if a_tokens_label is None:
            a_tokens_label = [t for t in a_tokens if not _is_value(t)]
        if b_tokens_label is None:
            b_tokens_label = [t for t in b_tokens if not _is_value(t)]

    sem        = tv.cosine([a_text_sem], [b_text_sem])[0][0]
    sem_labels = tv.cosine([a_text_lbl], [b_text_lbl])[0][0]

    sh  = 1.0 - norm_hamming(getattr(a, "simhash_text", 0), getattr(b, "simhash_text", 0))
    set_a, set_b = set(a_tokens), set(b_tokens)
    dice = _dice(set_a, set_b)
    chars = _char_similarity(a_text_clean, b_text_clean)
    try:
        na = getattr(a, "num_tokens", set())
        nb = getattr(b, "num_tokens", set())
        nums = (len(na & nb) / max(1, len(na | nb))) if (na or nb) else 0.0
    except Exception:
        nums = 0.0
    parent = 1.0 if getattr(a, "parent", "") == getattr(b, "parent", "") else 0.0
    page   = 1.0 - min(abs(getattr(a, "page_idx", 0) - getattr(b, "page_idx", 0))/50.0, 1.0)

    contain_tok    = _token_containment(a_tokens, b_tokens)
    contain_char   = _char_containment(a_text_clean, b_text_clean)
    contain_labels = _token_containment(a_tokens_label, b_tokens_label)

    A_anc, B_anc = set(getattr(a, "anchor_tokens", []) or []), set(getattr(b, "anchor_tokens", []) or [])
    inter = len(A_anc & B_anc); uni = len(A_anc | B_anc) or 1
    anchor_cover = (inter / max(1, len(A_anc))) if A_anc else 0.0
    anchor_jacc  = inter / uni
    A_sg, B_sg = set(getattr(a, "shingles4", []) or []), set(getattr(b, "shingles4", []) or [])
    sg_cover = (len(A_sg & B_sg) / max(1, len(A_sg))) if A_sg else 0.0

    w = {"sem":0.26, "sem_labels":0.10, "dice":0.15, "simhash":0.08, "chars":0.12, "nums":0.04,
         "parent":0.04, "page":0.02,
         "contain_tok":0.06, "contain_char":0.03, "contain_labels":0.04,
         "anchor_cover":0.03, "anchor_jacc":0.03, "shingle_cover":0.04}

    score = (w["sem"]*sem + w["sem_labels"]*sem_labels + w["dice"]*dice + w["simhash"]*sh + w["chars"]*chars +
             w["nums"]*nums + w["parent"]*parent + w["page"]*page +
             w["contain_tok"]*contain_tok + w["contain_char"]*contain_char + w["contain_labels"]*contain_labels +
             w["anchor_cover"]*anchor_cover + w["anchor_jacc"]*anchor_jacc + w["shingle_cover"]*sg_cover)

    sigs = {
        "semantic": sem, "sem_labels": sem_labels, "chars": chars,
        "contain_tok": contain_tok, "contain_char": contain_char, "contain_labels": contain_labels,
        "dice": dice, "simhash": sh, "nums": nums, "parent": parent, "page": page,
        "anchor_cover": anchor_cover, "anchor_jacc": anchor_jacc, "shingle_cover": sg_cover,
    }
    return score, sigs

def score_table(a, b, tv):
    # === canonical semantic table body ===
    a_txt = getattr(a, "table_semantic", None) or a.table_flat_text
    b_txt = getattr(b, "table_semantic", None) or b.table_flat_text

    # NEW: DID extraction (safe, only activates if present)
    # Use FULL text for DID extraction
    a_full = getattr(a, "raw_text", None) or getattr(a, "text", "") or a.table_flat_text
    b_full = getattr(b, "raw_text", None) or getattr(b, "text", "") or b.table_flat_text
    did_a = _extract_did(a_full)
    #print("DID A:", did_a)
    did_b = _extract_did(b_full)
    #print("DID B:", did_b)
    did_match = 1.0 if (did_a and did_b and did_a == did_b) else 0.0
    did_mismatch = 1.0 if (did_a and did_b and did_a != did_b) else 0.0

    sem = tv.cosine([a_txt], [b_txt])[0][0]
    sh    = 1.0 - norm_hamming(a.simhash_text, b.simhash_text)
    dice  = _dice(set(a.tokens), set(b.tokens))
    chars = _char_similarity(a_txt, b_txt)
    nums  = (len(a.num_tokens & b.num_tokens) / max(1, len(a.num_tokens | b.num_tokens))) if (a.num_tokens or b.num_tokens) else 0.0

    ## 25 02 2026 REGEX PREFIXES (e.g. "Table", "Fig", "Section") -- HELPS MATCHING STRUCTURAL ELEMENTS WITH SIMILAR ROLES
    def _reg_prefixes(tokens):
        out = set()
        for t in tokens:
            t = t.upper()
            if len(t) < 3:
                continue
            if any(c.isdigit() for c in t):
                prefix = ''.join([c for c in t if not c.isdigit()])
                prefix = prefix.strip("-_/")
                if 2 <= len(prefix) <= 10:
                    out.add(prefix)
        return out

    A_reg = _reg_prefixes(a.tokens)
    B_reg = _reg_prefixes(b.tokens)

    reg_jacc = len(A_reg & B_reg) / max(1, len(A_reg | B_reg))
    ## 25 02 2026 REGEX PREFIXES (e.g. "Table", "Fig", "Section") -- HELPS MATCHING STRUCTURAL ELEMENTS WITH SIMILAR ROLES

    teds    = teds_lite(a.table_sig or {}, b.table_sig or {})
    section = 1.0 if a.section_path==b.section_path else (0.5 if a.parent==b.parent else 0.0)

    contain_tok  = _token_containment(a.tokens, b.tokens)
    contain_char = _char_containment(a_txt, b_txt)

    A_anc, B_anc = set(a.anchor_tokens), set(b.anchor_tokens)
    inter = len(A_anc & B_anc); uni = len(A_anc | B_anc) or 1
    anchor_cover = (inter / max(1, len(A_anc))) if A_anc else 0.0
    anchor_jacc  = inter / uni
    A_sg, B_sg = set(a.shingles4), set(b.shingles4)
    sg_cover = (len(A_sg & B_sg) / max(1, len(A_sg))) if A_sg else 0.0

    header_j   = _jaccard_stringset(a.table_headers, b.table_headers)
    hdrbg_j    = _jaccard_stringset(a.header_bigrams, b.header_bigrams)
    rowbag_j   = _jaccard_stringset(a.row_sigs, b.row_sigs)

    # 24 03 2026 --- NEW: ROW OVERLAP (HARD SIGNAL) ---
    # --- IMPROVED: SOFT ROW OVERLAP ---
    def _normalize_row(r):
        txt = " ".join(r).lower()
        return " ".join(txt.split())

    def _row_similarity(a, b):
        # token jaccard
        ta = set(a.split())
        tb = set(b.split())
        inter = len(ta & tb)
        union = len(ta | tb) or 1
        jacc = inter / union

        # substring boost (handles OCR merged rows)
        if a in b or b in a:
            return max(jacc, 0.8)

        return jacc

    rowsA = [_normalize_row(r) for r in (a.grid_cells or [])[:50]]
    rowsB = [_normalize_row(r) for r in (b.grid_cells or [])[:50]]

    row_overlap = 0.0
    if rowsA and rowsB:
        matches = 0
        for ra in rowsA:
            best = 0
            for rb in rowsB:
                sim = _row_similarity(ra, rb)
                if sim > best:
                    best = sim

            # LOWER threshold (more tolerant)
            if best > 0.35:
                matches += 1

        row_overlap = matches / max(len(rowsA), 1)

    na = len(a.header_cols) or len(a.col_concat)
    nb = len(b.header_cols) or len(b.col_concat)
    col_count_sim = 1.0 - (abs(na - nb) / max(1, max(na, nb)))
    ra = len(a.grid_cells); rb = len(b.grid_cells)
    row_count_sim = 1.0 - (abs(ra - rb) / max(1, max(ra, rb)))

    def _col_sem_list(xcols: Dict[int,str], ycols: Dict[int,str]) -> List[float]:
        if not xcols or not ycols: return []
        xs = [xcols[k] for k in sorted(xcols.keys())]
        ys = [ycols[k] for k in sorted(ycols.keys())]
        M  = tv.cosine(xs, ys)
        flat = [v for row in M for v in row]
        flat.sort(reverse=True)
        return flat
    col_sims = _col_sem_list(a.col_concat, b.col_concat)
    col_top2 = sum(col_sims[:2]) / 2.0 if len(col_sims) >= 2 else (col_sims[0] if col_sims else 0.0)

    def _hungarian_like(sim_mat: List[List[float]]) -> float:
        if not sim_mat: return 0.0
        used=set(); sc=[]
        for i,row in enumerate(sim_mat):
            cand=[(row[j], j) for j in range(len(row)) if j not in used]
            if not cand: continue
            v,j=max(cand, key=lambda x:x[0])
            used.add(j); sc.append(max(0.0, min(1.0, v)))
        return sum(sc)/max(1,len(sc))

    def _col_numeric_sim(a_f, b_f, a_hdr: List[str], b_hdr: List[str]) -> float:
        aq = a_f.get("col_quant", {}); bq = b_f.get("col_quant", {})
        ascale = a_f.get("col_scale", {}); bscale = b_f.get("col_scale", {})
        amiss = a_f.get("col_missing_rate", {}); bmiss = b_f.get("col_missing_rate", {})
        A = sorted(aq.keys()); Bk = sorted(bq.keys())
        if not A or not Bk: return 0.0
        H = []
        if a_hdr and b_hdr:
            H = tv.cosine(a_hdr, b_hdr)
        S_num = []
        for i,ci in enumerate(A):
            row=[]
            for j,cj in enumerate(Bk):
                qsim = _cosine_list(aq.get(ci,[0]*5), bq.get(cj,[0]*5))
                scsim = 1.0 - min(1.0, abs(ascale.get(ci,-9)-bscale.get(cj,-9))/6.0)
                mrdiff = abs(amiss.get(ci,0.0)-bmiss.get(cj,0.0))
                misssim = 1.0 - min(1.0, mrdiff)
                if H and i < len(H) and j < len(H[i]):
                    hsim = H[i][j]
                    row.append(0.55*qsim + 0.20*scsim + 0.15*misssim + 0.10*hsim)
                else:
                    row.append(0.65*qsim + 0.25*scsim + 0.10*misssim)
            S_num.append(row)
        return _hungarian_like(S_num)

    col_quant_align = _col_numeric_sim(a.num_fingerprint, b.num_fingerprint, a.header_cols, b.header_cols)

    def _num_ratio_align(ar: Dict[int,float], br: Dict[int,float]) -> float:
        if not ar or not br: return 0.0
        I = sorted(ar.keys()); J = sorted(br.keys())
        C = [[1.0 - abs(ar[i] - br[j]) for j in J] for i in I]
        return _hungarian_like(C)

    type_align = _num_ratio_align(a.col_num_ratio, b.col_num_ratio)

    trend_sim = 0.0
    if a.num_fingerprint.get("has_index") and b.num_fingerprint.get("has_index"):
        Akeys = set(a.num_fingerprint.get("trend_sign", {}).keys())
        Bkeys = set(b.num_fingerprint.get("trend_sign", {}).keys())
        K = sorted(Akeys & Bkeys)
        if K:
            av = [a.num_fingerprint["trend_sign"][k] for k in K]
            bv = [b.num_fingerprint["trend_sign"][k] for k in K]
            trend_sim = _cosine_list(av, bv)

    corr_sim = _frobenius_sim(a.num_fingerprint.get("corr"), b.num_fingerprint.get("corr"))

    def _missing_vec(fp):
        mr = fp.get("col_missing_rate", {})
        keys = sorted(mr.keys())
        return [mr[k] for k in keys]
    miss_sim = _cosine_list(_missing_vec(a.num_fingerprint), _missing_vec(b.num_fingerprint))

    # w = {"sem":0.18, "dice":0.07, "simhash":0.05, "chars":0.08, "nums":0.04,
    #      "teds":0.05, "section":0.03,
    #      "contain_tok":0.03, "contain_char":0.02,
    #      "anchor_cover":0.03, "anchor_jacc":0.02, "shingle_cover":0.03,
    #      "header_j":0.05, "hdrbg_j":0.03, "rowbag_j":0.08, "col_top2":0.07,
    #      "col_count_sim":0.03, "row_count_sim":0.03,
    #      "type_align":0.06, "col_quant_align":0.09, "trend_sim":0.03, "corr_sim":0.04, "miss_sim":0.03}
    #"rowbag_j":0.08,
    w = {"sem":0.28, "dice":0.05, "simhash":0.03, "chars":0.06, "nums":0.04,
         "teds":0.05, "section":0.03,
         "contain_tok":0.03, "contain_char":0.02,
         "anchor_cover":0.03, "anchor_jacc":0.02, "shingle_cover":0.03,
         "header_j":0.05, "hdrbg_j":0.03, "rowbag_j":0.10, "col_top2":0.07,
         "col_count_sim":0.03, "row_count_sim":0.03,
         "type_align":0.06, "col_quant_align":0.09, "trend_sim":0.03, "corr_sim":0.04, "miss_sim":0.03,
         "reg_jacc": 0.05, "did_match": 0.12, "did_mismatch_penalty": -0.15,}

    # Latest night--- HARD STRUCTURE GUARD (prevents semantic leakage) ---
    structure_ok = (
        row_overlap >= 0.3 or
        rowbag_j >= 0.25 or
        col_top2 >= 0.5
    )

    if not structure_ok:
        sem *= 0.5
        col_top2 *= 0.6

    score = (w["sem"]*sem + w["dice"]*dice + w["simhash"]*sh + w["chars"]*chars +
             w["nums"]*nums + w["teds"]*teds + w["section"]*section +
             w["contain_tok"]*contain_tok + w["contain_char"]*contain_char +
             w["anchor_cover"]*anchor_cover + w["anchor_jacc"]*anchor_jacc + w["shingle_cover"]*sg_cover +
             w["header_j"]*header_j + w["hdrbg_j"]*hdrbg_j + w["rowbag_j"]*rowbag_j + w["col_top2"]*col_top2 +
             w["col_count_sim"]*col_count_sim + w["row_count_sim"]*row_count_sim +
             w["type_align"]*type_align + w["col_quant_align"]*col_quant_align + w["trend_sim"]*trend_sim +
             w["corr_sim"]*corr_sim + w["miss_sim"]*miss_sim+ w["reg_jacc"] * reg_jacc)
    
    # NEW: DID signal contribution (only active if DID present)
    score += w["did_match"] * did_match
    score += w["did_mismatch_penalty"] * did_mismatch

    # 24 03 2026--- NEW: APPLY ROW OVERLAP PENALTY/BOOST ---
    if row_overlap < 0.3:
        score *= 0.7

    if row_overlap > 0.6:
        score += 0.08
    sigs = {
        "semantic": sem, "chars": chars, "teds": teds, "section": section,
        "contain_tok": contain_tok, "contain_char": contain_char,
        "dice": dice, "simhash": sh, "nums": nums,
        "anchor_cover": anchor_cover, "anchor_jacc": anchor_jacc, "shingle_cover": sg_cover,
        "header_jacc": header_j, "header_bigram_jacc": hdrbg_j,
        "rowbag_jacc": rowbag_j, "col_sem_top2": col_top2,
        "col_count_sim": col_count_sim, "row_count_sim": row_count_sim,
        "type_align": type_align, "col_quant_align": col_quant_align,
        "trend_sim": trend_sim, "corr_sim": corr_sim, "miss_sim": miss_sim,
        "row_overlap": row_overlap, "reg_jacc": reg_jacc,
    }
    return score, sigs

# === Numeric-robust table scoring (gated wrapper) ===
def _infer_multiplier(h: str) -> float:
    s = (_norm_text(h or "")).replace("’", "'")
    if "('000" in s or "(000" in s:
        return 1_000.0
    if "million" in s:
        return 1_000_000.0
    if "billion" in s:
        return 1_000_000_000.0
    m = re.search(r"\(\s*0{3,}\s*[a-z]*\s*\)", s)
    if m:
        zeros = re.findall(r"0", m.group(0))
        return float(10 ** len(zeros))
    return 1.0

def _apply_col_multipliers(header_cols: list[str], grid_nums: list[list[Optional[float]]]) -> list[list[Optional[float]]]:
    if not header_cols or not grid_nums:
        return grid_nums
    mults = [ _infer_multiplier(h) for h in header_cols ]
    maxc = max((len(r) for r in grid_nums), default=0)
    out = []
    for r in grid_nums:
        row = r + [None]*(maxc-len(r))
        out_row = []
        for ci, v in enumerate(row):
            if v is None:
                out_row.append(None)
            else:
                m = mults[ci] if ci < len(mults) else 1.0
                out_row.append(v * m)
        out.append(out_row)
    return out

def _row_label_similarity(a_first: list[str], b_first: list[str]) -> float:
    def norm_cell(x: str) -> str:
        y = re.sub(r"\b\d+\b", " ", _norm_text(x or ""))
        return " ".join([t for t in y.split() if t])
    A = { norm_cell(x) for x in (a_first or []) if norm_cell(x) }
    B = { norm_cell(x) for x in (b_first or []) if norm_cell(x) }
    if not (A or B): return 0.0
    return len(A & B) / max(1, len(A | B))

def _dtw_sim(a: list[Optional[float]], b: list[Optional[float]]) -> float:
    xa = [x for x in a if x is not None]
    xb = [x for x in b if x is not None]
    if len(xa) < 3 or len(xb) < 3:
        return 0.0
    def znorm(xs):
        mu = statistics.fmean(xs)
        sd = statistics.pstdev(xs) or 1e-9
        return [(x - mu) / sd for x in xs]
    A = znorm(xa); B = znorm(xb)
    n, m = len(A), len(B)
    INF = 1e12
    D = [[INF]*(m+1) for _ in range(n+1)]
    D[0][0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = (A[i-1] - B[j-1])**2
            D[i][j] = cost + min(D[i-1][j], D[i][j-1], D[i-1][j-1])
    dist = D[n][m]/(n+m)
    return 1.0 / (1.0 + dist)

def _series_shape_similarity(a_grid: list[list[Optional[float]]],
                             b_grid: list[list[Optional[float]]],
                             a_idx_col: int, b_idx_col: int) -> float:
    if a_idx_col < 0 or b_idx_col < 0:
        return 0.0
    def cols_by_var(G):
        cols = _col_series(G)
        items = []
        for ci, xs in cols.items():
            vals = [v for v in xs if v is not None]
            if len(vals) >= 4:
                try:
                    var = statistics.pvariance(vals)
                except Exception:
                    var = 0.0
                items.append((var, ci))
        items.sort(reverse=True)
        return [ci for _,ci in items]
    a_top = cols_by_var(a_grid)[:3]
    b_top = cols_by_var(b_grid)[:3]
    if not a_top or not b_top:
        return 0.0
    sims = []
    colsA = _col_series(a_grid); colsB = _col_series(b_grid)
    for k in range(min(len(a_top), len(b_top))):
        ca, cb = a_top[k], b_top[k]
        sims.append(_dtw_sim(colsA.get(ca, []), colsB.get(cb, [])))
    if not sims:
        return 0.0
    return sum(sims)/len(sims)

def score_table_numeric(a, b, tv):
    a_grid = _apply_col_multipliers(getattr(a, "header_cols", []), getattr(a, "grid_nums", []))
    b_grid = _apply_col_multipliers(getattr(b, "header_cols", []), getattr(b, "grid_nums", []))

    a_fp = getattr(a, "num_fingerprint", {}) or _build_numeric_fingerprint(a_grid)
    b_fp = getattr(b, "num_fingerprint", {}) or _build_numeric_fingerprint(b_grid)

    def _hungarian_like(sim_mat: List[List[float]]) -> float:
        if not sim_mat: return 0.0
        used=set(); sc=[]
        for i,row in enumerate(sim_mat):
            cand=[(row[j], j) for j in range(len(row)) if j not in used]
            if not cand: continue
            v,j=max(cand, key=lambda x:x[0])
            used.add(j); sc.append(max(0.0, min(1.0, v)))
        return sum(sc)/max(1,len(sc))

    def _col_numeric_sim(a_f, b_f, a_hdr: List[str], b_hdr: List[str]) -> float:
        aq = a_f.get("col_quant", {}); bq = b_f.get("col_quant", {})
        ascale = a_f.get("col_scale", {}); bscale = b_f.get("col_scale", {})
        amiss = a_f.get("col_missing_rate", {}); bmiss = b_f.get("col_missing_rate", {})
        A = sorted(aq.keys()); Bk = sorted(bq.keys())
        if not A or not Bk: return 0.0
        H = []
        if a_hdr and b_hdr:
            H = tv.cosine(a_hdr, b_hdr)
        S_num = []
        for i,ci in enumerate(A):
            row=[]
            for j,cj in enumerate(Bk):
                qsim = _cosine_list(aq.get(ci,[0]*5), bq.get(cj,[0]*5))
                scsim = 1.0 - min(1.0, abs(ascale.get(ci,-9)-bscale.get(cj,-9))/6.0)
                mrdiff = abs(amiss.get(ci,0.0)-bmiss.get(cj,0.0))
                misssim = 1.0 - min(1.0, mrdiff)
                if H:
                    hsim = H[i][j]
                    row.append(0.60*qsim + 0.20*scsim + 0.15*misssim + 0.05*hsim)
                else:
                    row.append(0.70*qsim + 0.30*scsim)
            S_num.append(row)
        return _hungarian_like(S_num)

    col_quant_align = _col_numeric_sim(a_fp, b_fp, getattr(a, "header_cols", []), getattr(b, "header_cols", []))

    def _num_ratio_align(ar: Dict[int,float], br: Dict[int,float]) -> float:
        if not ar or not br: return 0.0
        I = sorted(ar.keys()); J = sorted(br.keys())
        C = [[1.0 - abs(ar[i] - br[j]) for j in J] for i in I]
        return _hungarian_like(C)

    type_align = _num_ratio_align(getattr(a, "col_num_ratio", {}), getattr(b, "col_num_ratio", {}))
    row_label_j = _row_label_similarity(getattr(a, "table_first_col", []), getattr(b, "table_first_col", []))

    trend_sim = 0.0
    if a_fp.get("has_index") and b_fp.get("has_index"):
        Akeys = set(a_fp.get("trend_sign", {}).keys()) & set(b_fp.get("trend_sign", {}).keys())
        if Akeys:
            av = [a_fp["trend_sign"][k] for k in sorted(Akeys)]
            bv = [b_fp["trend_sign"][k] for k in sorted(Akeys)]
            trend_sim = _cosine_list(av, bv)

    series_shape = _series_shape_similarity(a_grid, b_grid, a_fp.get("index_col",-1), b_fp.get("index_col",-1))

    na = len(getattr(a, "header_cols", [])) or len(getattr(a, "col_concat", {}))
    nb = len(getattr(b, "header_cols", [])) or len(getattr(b, "col_concat", {}))
    col_count_sim = 1.0 - (abs(na - nb) / max(1, max(na, nb)))
    ra = len(getattr(a, "grid_cells", [])); rb = len(getattr(b, "grid_cells", []))
    row_count_sim = 1.0 - (abs(ra - rb) / max(1, max(ra, rb)))

    w = {
        "col_quant_align":0.38, "type_align":0.12, "row_label":0.12,
        "trend":0.10, "series":0.16, "col_count":0.06, "row_count":0.06
    }
    score = (w["col_quant_align"]*col_quant_align +
             w["type_align"]*type_align +
             w["row_label"]*row_label_j +
             w["trend"]*trend_sim +
             w["series"]*series_shape +
             w["col_count"]*col_count_sim +
             w["row_count"]*row_count_sim)

    sigs = {
        "num_col_quant_align": col_quant_align,
        "num_type_align": type_align,
        "num_row_label_jacc": row_label_j,
        "num_trend_sim": trend_sim,
        "num_series_shape": series_shape,
        "num_col_count_sim": col_count_sim,
        "num_row_count_sim": row_count_sim
    }
    return score, sigs

def score_table_robust(a, b, tv):
    base_score, base_sigs = score_table(a, b, tv)
    a_numdom = sum(getattr(a, "col_num_ratio", {}).values())/max(1, len(getattr(a, "col_num_ratio", {}))) if getattr(a, "col_num_ratio", {}) else 0.0
    b_numdom = sum(getattr(b, "col_num_ratio", {}).values())/max(1, len(getattr(b, "col_num_ratio", {}))) if getattr(b, "col_num_ratio", {}) else 0.0
    if a_numdom < 0.55 or b_numdom < 0.55:
        base_sigs = dict(base_sigs); base_sigs["path"] = "base"
        return base_score, base_sigs

    num_score, num_sigs = score_table_numeric(a, b, tv)
    if num_score > base_score:
        out_sigs = dict(base_sigs); out_sigs.update(num_sigs); out_sigs["path"] = "numeric"
        return num_score, out_sigs
    else:
        base_sigs = dict(base_sigs); base_sigs["path"] = "base"
        return base_score, base_sigs

def score_image(a, b, tv):
    sem = tv.cosine([a.text_clean], [b.text_clean])[0][0]
    sh  = 1.0 - norm_hamming(a.simhash_text, b.simhash_text)
    dice = _dice(set(a.tokens), set(b.tokens))
    chars = _char_similarity(a.text_clean, b.text_clean)
    nums = (len(a.num_tokens & b.num_tokens) / max(1, len(a.num_tokens | b.num_tokens))) if (a.num_tokens or b.num_tokens) else 0.0
    section = 1.0 if a.section_path==b.section_path else (0.5 if a.parent==b.parent else 0.0)

    contain_tok  = _token_containment(a.tokens, b.tokens)
    contain_char = _char_containment(a.text_clean, b.text_clean)

    A_anc, B_anc = set(a.anchor_tokens), set(b.anchor_tokens)
    inter = len(A_anc & B_anc); uni = len(A_anc | B_anc) or 1
    anchor_cover = (inter / max(1, len(A_anc))) if A_anc else 0.0
    anchor_jacc  = inter / uni
    A_sg, B_sg = set(a.shingles4), set(b.shingles4)
    sg_cover = (len(A_sg & B_sg) / max(1, len(A_sg))) if A_sg else 0.0

    w = {"sem":0.38, "dice":0.18, "simhash":0.10, "chars":0.14, "nums":0.05,
         "section":0.03, "contain_tok":0.04, "contain_char":0.02,
         "anchor_cover":0.04, "anchor_jacc":0.02, "shingle_cover":0.04}

    score = (w["sem"]*sem + w["dice"]*dice + w["simhash"]*sh + w["chars"]*chars +
             w["nums"]*nums + w["section"]*section +
             w["contain_tok"]*contain_tok + w["contain_char"]*contain_char +
             w["anchor_cover"]*anchor_cover + w["anchor_jacc"]*anchor_jacc +
             w["shingle_cover"]*sg_cover)

    sigs = {
        "semantic": sem, "chars": chars, "section": section,
        "contain_tok": contain_tok, "contain_char": contain_char,
        "dice": dice, "simhash": sh, "nums": nums,
        "anchor_cover": anchor_cover, "anchor_jacc": anchor_jacc, "shingle_cover": sg_cover,
    }
    return score, sigs

# ---------- Hungarian ----------
def hungarian(cost_matrix: List[List[float]]) -> List[Tuple[int,int]]:
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as _np
        cm = _np.array(cost_matrix)
        r, c = linear_sum_assignment(cm)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        pass
    res=[]; used_cols=set()
    for i,row in enumerate(cost_matrix):
        cand=[(v,j) for j,v in enumerate(row) if j not in used_cols]
        if not cand: continue
        v,j=min(cand, key=lambda x:x[0])
        used_cols.add(j); res.append((i,j))
    return res

# ---------- Build artifacts ----------
class _TmpArtifact(Artifact):  # not used; kept for typing clarity
    pass

def build_artifacts(data: List[Dict[str,Any]],
                    image_dir: str = ".",
                    stopset: Optional[Set[str]] = None,
                    idf_map: Optional[Dict[str,float]] = None) -> List[Artifact]:
    if idf_map is None:
        texts = [e.get("text","") for e in data if e.get("type") in ("table","image","text")]
        idf_map = _idf_from_texts(texts)

    arts: List[Artifact] = []
    for idx, e in enumerate(data):
        t = e.get("type","text").lower()
        if t not in ("table","image","text"): continue
        aid = f"{t}-{idx}"
        text_content = e.get("text","")
        img_path = e.get("img_path")
        if isinstance(img_path, str) and img_path.strip():
            basename = os.path.basename(img_path).lstrip("./\\")
            img_path = os.path.join("images", basename).replace("\\", "/")
        else:
            img_path = None
        html_body = e.get("table_body") if t=="table" else None
        order = e.get("order", idx)
        art = Artifact(
            aid, t, e.get("page_idx",-1), e.get("parent",""),
            text_content, e.get("source_file",""),
            img_path=img_path, html=html_body, stopset=stopset, idf_map=idf_map, order=order
        )
        arts.append(art)
    return arts