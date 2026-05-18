import os, json, pandas as pd, numpy as np, re, html, math, collections
import traceback
from typing import List, Dict, Tuple, Any, Set, Optional
from app.delta_comparator.utils.logger import log as logging
from pathlib import Path
import zipfile
import shutil
import tempfile
import asyncio
import functools
import time
import concurrent.futures
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from typing import List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from rabbitMQ_manager import create_queue_publish
from app.delta_comparator.core.match_pipeline import match_revision_image_first
from app.delta_comparator.core.logging_setup import configure_stdout_logging

configure_stdout_logging("INFO")
try:
    from bs4 import BeautifulSoup
    _HAVE_BS4 = True
except Exception:
    _HAVE_BS4 = False

_NUM_RE = re.compile(r"\b\d+(?:[./]\d+)*\b")
# ----------------------------
# Logging
# ----------------------------
# logging.remove()
# logging.add(lambda msg: print(msg, end=""), level="INFO")
# logging.add("match.log", level="DEBUG", rotation="2 MB", enqueue=True)

# ----------------------------
# Config
# ----------------------------
IMAGE_DIR = "Images/"

# Provide as many versions as you like (ordered old -> new)

THRESHOLDS = {
    "table": 0.15,
    "image": 0.55,  # prepass uses image_threshold
    "text":  0.35,
}
POLICY = {
    "min_margin": 0.05,
    "parent_penalty": 0.05,
    "page_distance_soft": 40,
    "page_penalty": 0.05,
    "min_signals": {
        "table": {"teds": 0.06},
        "text":  {"chars_or_sem": 0.35, "containment": 0.35},
    }
}

# Choose: "all" | "table" | "image" | "text" | list like ["table","text"]

# ===== New knobs to help numeric-heavy tables =====
# Allow the same revised table to be used as a NEAR suggestion for multiple source tables.
# Accepted matches remain strictly one-to-one.
ALLOW_NEAR_REUSE = True

# Coalescing merges “near-identical” table rows across versions.
# - "auto": coalesce only when NOT numeric-heavy (safe)
# - "on":   always coalesce
# - "off":  never coalesce
TABLE_COALESCE_MODE = "auto"

#==============ADDED To detect footer tables===========
FOOTER_TABLE_KEYWORDS = {
    "product group",
    "date/compiler",
    "product identification",
    "doc.code",
    "doc code",
    "page",
    "wabco"
}

def is_footer_table_html(html: str) -> bool:
    if not html:
        return False
    t = html.lower()
    if "<table" not in t:
        return False
    return sum(1 for k in FOOTER_TABLE_KEYWORDS if k in t) >= 3
#==============ADDED To detect footer tables===========

# =========================================================
# Small helpers
# =========================================================
def _suffix_from_filter(f):
    if isinstance(f, str):
        return f.lower()
    try:
        return "_".join(sorted(t.lower() for t in f))
    except Exception:
        return "all"

# =========================================================
# Preprocessing: handle zip/json inputs
# =========================================================
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

def _safe_makedirs(p: Path):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _normalize_type_filter(type_filter):
    allowed = {"image", "table", "text"}
    if type_filter is None or str(type_filter).lower() == "all":
        return allowed
    if isinstance(type_filter, str):
        return {type_filter.strip().lower()} & allowed
    try:
        return set(t.strip().lower() for t in type_filter) & allowed
    except Exception:
        return allowed

def _is_blank_cell(val) -> bool:
    if val is None:
        return True
    try:
        if isinstance(val, float) and math.isnan(val):
            return True
    except Exception:
        pass
    if isinstance(val, str):
        return len(val.strip()) == 0
    if isinstance(val, (list, tuple, set, dict)):
        return len(val) == 0
    try:
        import numpy as _np
        if isinstance(val, _np.ndarray):
            return val.size == 0
    except Exception:
        pass

def _norm_img_path(p: Any) -> Any:
    if not p or not isinstance(p, str): return p
    base = os.path.basename(p).lstrip("./\\")
    return os.path.join("images", base).replace("\\", "/")

def _mk_pair_col(i, j, base):  # i and j are 1-based version indices
    return f"{base}_{i}_{j}"


_MATCH_KINDS = ("image", "table", "text")
_ARTIFACT_KEEP_FIELDS = (
    "id",
    "type",
    "text",
    "img_path",
    "table_body",
    "page_idx",
    "order",
    "parent",
)
_MATCH_KEEP_FIELDS = (
    "source_id",
    "revision_id",
    "score",
    "signals",
    "source_meta",
    "revision_meta",
    "source_text",
    "revision_text",
    "source_img_path",
    "revision_img_path",
    "source_table_body",
    "revision_table_body",
)


def _trim_artifact_payload(artifacts: Dict[str, dict]) -> Dict[str, dict]:
    trimmed = {}
    for artifact_id, artifact in (artifacts or {}).items():
        trimmed[artifact_id] = {
            field: artifact.get(field)
            for field in _ARTIFACT_KEEP_FIELDS
        }
    return trimmed


def _trim_match_payload(matches: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    trimmed = {}
    for kind in _MATCH_KINDS:
        rows = (matches or {}).get(kind)
        if not rows:
            continue
        trimmed[kind] = [
            {field: row.get(field) for field in _MATCH_KEEP_FIELDS}
            for row in rows
        ]
    return trimmed


def _trim_pair_result(res: dict) -> dict:
    artifacts = res.get("artifacts", {})
    return {
        "matches": _trim_match_payload(res.get("matches", {})),
        "near_matches": _trim_match_payload(res.get("near_matches", {})),
        "artifacts": {
            "source": _trim_artifact_payload(artifacts.get("source", {})),
            "revision": _trim_artifact_payload(artifacts.get("revision", {})),
        },
    }

# =========================================================
# Table signature utilities (numeric-agnostic coalescing)
# =========================================================

def _tok_label_only(s: str) -> list:
    s = html.unescape(str(s or ""))
    s = re.sub(r"<\s*br\s*/?\s*>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("|", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    toks = re.findall(r"[a-z0-9%/+\-\.]+", s)
    out = []
    for t in toks:
        if _NUM_RE.fullmatch(t):
            continue
        out.append(t)
    return out

def _numeric_density(text: str) -> float:
    if not text: return 0.0
    tokens = re.findall(r"[^\s|]+", text)
    if not tokens: return 0.0
    n_nums = sum(1 for t in tokens if _NUM_RE.search(t))
    return n_nums / max(1, len(tokens))

def _extract_md_headers(md: str) -> Tuple[Set[str], int, int]:
    lines = [ln.strip() for ln in (md or "").splitlines() if ln.strip()]
    pipe_lines = [ln for ln in lines if "|" in ln]
    if not pipe_lines:
        return set(), 0, 0
    header_line = pipe_lines[0]
    hdr = re.sub(r"^\|?(.+?)\|?$", r"\1", header_line)
    headers = [c.strip() for c in hdr.split("|")]
    toks = set()
    for h in headers:
        toks |= set(_tok_label_only(h))
    data_lines = [ln for ln in pipe_lines[1:] if not re.fullmatch(r"[:\-\s\|]+", ln)]
    rows = len(data_lines)
    cols = max(
        len(re.sub(r"^\|?(.+?)\|?$", r"\1", ln).split("|"))
        for ln in [header_line] + data_lines
    ) if pipe_lines else 0
    return toks, rows, cols

def _extract_html_headers(ht: str) -> Tuple[Set[str], int, int]:
    if not _HAVE_BS4:
        return set(), 0, 0
    try:
        soup = BeautifulSoup(ht or "", "lxml")
        tbl = soup.find("table")
        if not tbl:
            return set(), 0, 0
        ths = tbl.find_all("th")
        toks = set()
        for th in ths:
            toks |= set(_tok_label_only(th.get_text(" ", strip=True)))
        rows = len(tbl.find_all("tr"))
        max_cols = 0
        for tr in tbl.find_all("tr"):
            c = 0
            for cell in tr.find_all(["td", "th"]):
                span = int(cell.get("colspan", "1") or 1)
                c += span
            max_cols = max(max_cols, c)
        return toks, rows, max_cols
    except Exception:
        return set(), 0, 0

def table_signature(blob: str, fallback_text: str = "") -> str:
    s = blob or ""
    headers: Set[str] = set()
    rows = cols = 0
    low = (s or "").lower()
    if "<table" in low or "</tr>" in low:
        h, r, c = _extract_html_headers(s)
        headers |= h; rows, cols = r, c
    elif "|" in s:
        h, r, c = _extract_md_headers(s)
        headers |= h; rows, cols = r, c
    if not headers and fallback_text:
        headers |= set(_tok_label_only(fallback_text))
    hdr_sorted = sorted(list(headers))[:20]
    hdr_part = " ".join(hdr_sorted)
    rb = 0 if rows <= 0 else (1 if rows <= 5 else (2 if rows <= 10 else 3))
    cb = 0 if cols <= 0 else (1 if cols <= 4 else (2 if cols <= 8 else 3))
    return f"h:{hdr_part} | rbin:{rb} cbin:{cb}"

# =========================================================
# Pairwise matching
# =========================================================
def run_pairwise_matches(
    json_paths: List[str],
    combined_images,
    run_kinds="all",
    task_id: str | None = None,
    progress_sink=None,
) -> List[dict]:
    results = []
    for i in range(len(json_paths)-1):
        with open(json_paths[i], "r", encoding="utf-8") as f:
            A_raw = json.load(f)
        with open(json_paths[i+1], "r", encoding="utf-8") as f:
            B_raw = json.load(f)

        #logging.debug(f"Calling ***** Using combined images dir: {combined_images}")
        res = match_revision_image_first(
            A_raw, B_raw,
            image_dir=combined_images,
            thresholds=THRESHOLDS,
            policy=POLICY,
            image_threshold=0.25,
            shortlist_k=24,
            run_kinds=run_kinds,
            task_id=task_id,
            progress_sink=progress_sink,
        )
        results.append(res)
        #logging.debug(f"Pair v{i+1}-v{i+2}")
        img_count = len(res['matches']["image"])
        table_count = len(res["matches"]["table"])
        text_count = len(res["matches"]["table"])
        #logging.debug(f"matched image= {img_count}" ,)
        #logging.debug(f'table=  {table_count}')
        #logging.debug(f'Text= {text_count}')
    return results


def _match_pair_worker(payload: Tuple[int, list, list, str, Any]) -> Tuple[int, dict]:
    """
    Process-safe worker for one adjacent pair.
    Returns (pair_idx, result) so caller can preserve deterministic order.
    """
    pair_idx, A_raw, B_raw, combined_images, run_kinds = payload
    res = match_revision_image_first(
        A_raw,
        B_raw,
        image_dir=combined_images,
        thresholds=THRESHOLDS,
        policy=POLICY,
        image_threshold=0.25,
        shortlist_k=24,
        run_kinds=run_kinds,
        task_id=None,
        progress_sink=None,
    )
    return pair_idx, _trim_pair_result(res)

def _match_pair_worker_from_paths(payload: Tuple[int, str, str, str, Any]) -> Tuple[int, dict]:
    """
    Process-safe worker for one adjacent pair that reads JSON files inside the worker.
    This avoids pickling large Python objects across process boundaries.
    """
    pair_idx, a_path, b_path, combined_images, run_kinds = payload
    with open(a_path, "r", encoding="utf-8") as fa:
        A_raw = json.load(fa)
    with open(b_path, "r", encoding="utf-8") as fb:
        B_raw = json.load(fb)

    res = match_revision_image_first(
        A_raw,
        B_raw,
        image_dir=combined_images,
        thresholds=THRESHOLDS,
        policy=POLICY,
        image_threshold=0.25,
        shortlist_k=24,
        run_kinds=run_kinds,
        task_id=None,
        progress_sink=None,
    )
    return pair_idx, _trim_pair_result(res)

def collect_version_artifacts(pair_results: List[dict], nV: int, kinds: Set[str]) -> List[Dict[str,dict]]:
    version_lookup = [collections.defaultdict(dict) for _ in range(nV)]
    for i, res in enumerate(pair_results):
        artifacts = res.get("artifacts", {})
        source_artifacts = artifacts.get("source", {})
        revision_artifacts = artifacts.get("revision", {})
        for aid, a in source_artifacts.items():
            if a.get("type") in kinds:
                version_lookup[i][aid] = a
        for bid, b in revision_artifacts.items():
            if b.get("type") in kinds:
                version_lookup[i+1][bid] = b
    return [dict(d) for d in version_lookup]

# =========================================================
# Edge building (accepted + best near; keep scores)
# =========================================================
def _best_one_to_one_edges(res: dict, kind: str) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Returns:
      accepted_map: sid -> {revision_id, score, ... , is_near=0}
      near_map    : sid -> {revision_id, score, ... , is_near=1}  # only when sid not in accepted_map
    Enforces 1-1 by picking highest score per revision_id among matches (ACCEPTED).
    NEAR suggestions do NOT enforce 1-1 (controlled later via ALLOW_NEAR_REUSE).
    """
    cand_by_rev = {}
    for m in res["matches"].get(kind, []):
        sid, rid, score = m["source_id"], m["revision_id"], float(m.get("score", 0.0))
        prev = cand_by_rev.get(rid)
        if prev is None or score > prev["score"]:
            cand_by_rev[rid] = {"source_id": sid, "m": m, "score": score}

    accepted_map: Dict[str, dict] = {}
    for rid, item in cand_by_rev.items():
        sid = item["source_id"]
        m   = item["m"]
        accepted_map[sid] = {
            "revision_id": m["revision_id"],
            "score": float(m.get("score", 0.0)),
            "signals": m.get("signals", {}),
            "source_meta": m.get("source_meta", {}),
            "revision_meta": m.get("revision_meta", {}),
            "source_text": m.get("source_text"),
            "revision_text": m.get("revision_text"),
            "source_img_path": m.get("source_img_path"),
            "revision_img_path": m.get("revision_img_path"),
            "source_table_body": m.get("source_table_body"),
            "revision_table_body": m.get("revision_table_body"),
            "is_near": 0,
        }

    # NEAR: best per source_id that wasn't accepted (no 1-1 restriction here)
    # Replace NEAR section with this (no sort)
    best_near = {}

    for nm in res.get("near_matches", {}).get(kind, []):
        sid = nm["source_id"]
        if sid in accepted_map:
            continue

        score = float(nm.get("score", 0.0))
        prev = best_near.get(sid)

        if prev is None or score > prev["score"]:
            best_near[sid] = {
                "revision_id": nm["revision_id"],
                "score": score,
                "signals": nm.get("signals", {}),
                "source_meta": nm.get("source_meta", {}),
                "revision_meta": nm.get("revision_meta", {}),
                "source_text": nm.get("source_text"),
                "revision_text": nm.get("revision_text"),
                "source_img_path": nm.get("source_img_path"),
                "revision_img_path": nm.get("revision_img_path"),
                "source_table_body": nm.get("source_table_body"),
                "revision_table_body": nm.get("revision_table_body"),
                "is_near": 1,
            }
    return accepted_map, best_near

def build_edges_all_pairs(pair_results: List[dict], kinds: Set[str]) -> Dict[str, List[Tuple[Dict[str,dict], Dict[str,dict]]]]:
    edges = {k: [] for k in kinds}
    for res in pair_results:
        for k in kinds:
            acc, near = _best_one_to_one_edges(res, k)
            edges[k].append((acc, near))
    return edges

def _parent_same(metaA: dict, metaB: dict) -> Any:
    if not metaA or not metaB: return None
    try:
        return int((metaA.get("parent") or "") == (metaB.get("parent") or ""))
    except Exception:
        return None

# =========================================================
# Chaining with strict order preservation + near usage
# =========================================================
def _order_key_from_artifact(a: dict) -> Tuple[int, int]:
    # Smaller (page_idx, order) first; missing → large
    so = a.get("order")
    pg = a.get("page_idx")
    if so is None: so = np.iinfo(np.int32).max
    if pg is None: pg = np.iinfo(np.int32).max
    return (pg, so)

def _chain_for_kind(
    kind: str,
    nV: int,
    version_artifacts: List[Dict[str,dict]],
    edges_for_kind: List[Tuple[Dict[str,dict], Dict[str,dict]]],
) -> List[dict]:
    """
    Build row chains for one type across N versions using accepted edges (and near if no accepted).
    STRICT order: iterate v1 in natural order → v2 leftovers → v3 leftovers → ...
    """
    # Index version artifacts by id & get ordered lists per version
    V: List[Dict[str, dict]] = []
    V_ordered_ids: List[List[str]] = []

    for vi in range(nV):
        vmap_src = version_artifacts[vi]
        vmap = {aid: a for aid, a in vmap_src.items() if a["type"] == kind}
        V.append(vmap)
        ordered_ids = sorted(
            vmap.keys(),
            key=lambda _id: _order_key_from_artifact(vmap[_id])
        )
        V_ordered_ids.append(ordered_ids)

    # Forward edges per pair: sid -> edge_info (accepted); near sid -> edge_info
    E = [{} for _ in range(nV - 1)]
    ENEAR = [{} for _ in range(nV - 1)]
    for i in range(nV - 1):
        acc, near = edges_for_kind[i]
        E[i] = acc
        ENEAR[i] = near

    visited = set()
    rows_out: List[dict] = []
    chain_id = 0

    # Reused templates reduce per-row object churn.
    version_template = {}
    for vi in range(nV):
        version_template[f"v{vi+1}_id"] = None
        version_template[f"v{vi+1}_text"] = None
        version_template[f"v{vi+1}_img_path"] = None
        version_template[f"v{vi+1}_table_body"] = None
        version_template[f"v{vi+1}_page"] = None
        version_template[f"v{vi+1}_order"] = None

    edge_template = {}
    pair_cols = []
    for vi in range(nV - 1):
        score_col = _mk_pair_col(vi+1, vi+2, "score")
        is_near_col = _mk_pair_col(vi+1, vi+2, "is_near")
        parent_same_col = _mk_pair_col(vi+1, vi+2, "parent_same")
        imgsim_col = _mk_pair_col(vi+1, vi+2, "imgsim")
        textsim_col = _mk_pair_col(vi+1, vi+2, "textsim")
        ssim_col = _mk_pair_col(vi+1, vi+2, "ssim")

        pair_cols.append((score_col, is_near_col, parent_same_col, imgsim_col, textsim_col, ssim_col))
        edge_template[score_col] = None
        edge_template[is_near_col] = None
        edge_template[parent_same_col] = None
        edge_template[imgsim_col] = None
        edge_template[textsim_col] = None
        edge_template[ssim_col] = None

    for start_vi in range(nV):
        for sid in V_ordered_ids[start_vi]:
            if (start_vi, sid) in visited:
                continue

            chain_id += 1
            row = {
                "type": kind,
                "chain_id": chain_id,
                "pair_index": f"v{start_vi+1}-v{nV}",
            }
            row.update(version_template)
            row.update(edge_template)

            cur_vi, cur_id = start_vi, sid
            anchor_vi: Optional[int] = None
            anchor_page = 10**9
            anchor_order = 10**9

            while True:
                vmap = V[cur_vi]
                a = vmap.get(cur_id)
                if a:
                    row[f"v{cur_vi+1}_id"] = cur_id
                    row[f"v{cur_vi+1}_text"] = a.get("text")
                    row[f"v{cur_vi+1}_img_path"] = _norm_img_path(a.get("img_path"))
                    row[f"v{cur_vi+1}_table_body"] = a.get("table_body")
                    row[f"v{cur_vi+1}_page"] = a.get("page_idx")
                    row[f"v{cur_vi+1}_order"] = a.get("order")
                    if anchor_vi is None:
                        anchor_vi = cur_vi
                        anchor_page = a.get("page_idx") if a.get("page_idx") is not None else 10**9
                        anchor_order = a.get("order") if a.get("order") is not None else 10**9

                if cur_vi >= nV - 1:
                    visited.add((cur_vi, cur_id))
                    break

                emap = E[cur_vi]
                edge = emap[cur_id] if cur_id in emap else None
                used_near = False
                if edge is None:
                    nmap = ENEAR[cur_vi]
                    edge = nmap[cur_id] if cur_id in nmap else None
                    used_near = edge is not None

                if edge is None:
                    visited.add((cur_vi, cur_id))
                    break

                rid = edge["revision_id"]
                score_col, is_near_col, parent_same_col, imgsim_col, textsim_col, ssim_col = pair_cols[cur_vi]
                row[score_col] = edge.get("score")
                row[is_near_col] = edge.get("is_near", 0)
                right_art = V[cur_vi + 1].get(rid)
                row[parent_same_col] = _parent_same(a, right_art)
                sig = edge.get("signals") or {}
                if sig.get("img_sim") is not None:
                    row[imgsim_col] = float(sig["img_sim"])
                if sig.get("text_sim") is not None:
                    row[textsim_col] = float(sig["text_sim"])
                if sig.get("ssim") is not None:
                    row[ssim_col] = float(sig["ssim"])

                visited.add((cur_vi, cur_id))
                if not used_near or not ALLOW_NEAR_REUSE:
                    visited.add((cur_vi + 1, rid))
                cur_vi += 1
                cur_id = rid

            row["_anchor_vi"] = anchor_vi if anchor_vi is not None else nV
            row["_anchor_page"] = anchor_page
            row["_anchor_order"] = anchor_order
            rows_out.append(row)
    return rows_out

# =========================================================
# Build N-version DataFrame
# =========================================================

def build_n_version_dataframe(json_paths: List[str], combined_images, type_filter="all", task_id: str | None = None, progress_sink=None) -> pd.DataFrame:
    try:
        kinds = _normalize_type_filter(type_filter)
        nV = len(json_paths)
        assert nV >= 2, "Provide at least two versions."

        # -------------------------------
        # Optimized Pairwise Matching
        # -------------------------------
        t_pair_start = time.perf_counter()

        pair_count = nV - 1
        pair_results = [None] * pair_count

        pair_args_paths = [
            (i, json_paths[i], json_paths[i + 1], combined_images, type_filter)
            for i in range(pair_count)
        ]


        if pair_count == 1:
            logging.info("Single pair detected -> running in-process")
            idx, a_path, b_path, _, run_kinds = pair_args_paths[0]
            with open(a_path, "r", encoding="utf-8") as fa:
                A_raw = json.load(fa)
            with open(b_path, "r", encoding="utf-8") as fb:
                B_raw = json.load(fb)

            res = match_revision_image_first(
                A_raw,
                B_raw,
                image_dir=combined_images,
                thresholds=THRESHOLDS,
                policy=POLICY,
                image_threshold=0.25,
                shortlist_k=24,
                run_kinds=run_kinds,
                task_id=task_id,
                progress_sink=progress_sink,
            )

            pair_results[idx] = _trim_pair_result(res)
            del res

        else:
            executor_cls = ProcessPoolExecutor

            # 🔥 LIMIT WORKERS (CRITICAL FOR MEMORY)
            max_workers = min(pair_count, 4)

            logging.info(f"Using ProcessPoolExecutor with {max_workers} workers")

            with executor_cls(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_match_pair_worker_from_paths, args)
                    for args in pair_args_paths
                ]

                for future in as_completed(futures):
                    idx, res = future.result()

                    pair_results[idx] = res

                    del res  # 🔥 FREE MEMORY

        if any(result is None for result in pair_results):
            raise RuntimeError("Pairwise mapping did not complete for all version pairs")

        logging.info(f"[PERF] pair_matching took {(time.perf_counter() - t_pair_start):.3f}s")
        # -------------------------------
        # Artifact + Edge Building
        # -------------------------------
        t_collect_start = time.perf_counter()

        version_artifacts = collect_version_artifacts(pair_results, nV, kinds)
        edges_all = build_edges_all_pairs(pair_results, kinds)

        logging.info(f"[PERF] artifact_collection and edge_building took {(time.perf_counter() - t_collect_start):.3f}s")

        # -------------------------------
        # Chaining
        # -------------------------------
        t_chain_start = time.perf_counter()

        all_rows = []
        for kind in sorted(kinds):
            rows = _chain_for_kind(kind, nV, version_artifacts, edges_all[kind])
            all_rows.extend(rows)

        logging.info(f"[PERF] chaining took {(time.perf_counter() - t_chain_start):.3f}s")

        # -------------------------------
        # DataFrame Creation
        # -------------------------------
        t_frame_start = time.perf_counter()

        df = pd.DataFrame.from_records(all_rows)

        if not df.empty:
            df = df.sort_values(
                by=["_anchor_vi", "_anchor_page", "_anchor_order", "chain_id"],
                ascending=[True, True, True, True],
                kind="mergesort"
            ).reset_index(drop=True)

        # normalize image paths
        for vi in range(nV):
            col = f"v{vi+1}_img_path"
            if col in df.columns:
                df[col] = df[col].apply(_norm_img_path)

        # cleanup
        df = df.drop(
            columns=[c for c in df.columns if c.startswith("_anchor_")],
            errors="ignore"
        )

        # column ordering
        cols = ["type", "chain_id", "pair_index"]
        for vi in range(nV):
            cols += [
                f"v{vi+1}_id",
                f"v{vi+1}_text",
                f"v{vi+1}_img_path",
                f"v{vi+1}_table_body",
                f"v{vi+1}_page",
                f"v{vi+1}_order",
            ]

        for vi in range(nV - 1):
            cols += [
                _mk_pair_col(vi+1, vi+2, "score"),
                _mk_pair_col(vi+1, vi+2, "is_near"),
                _mk_pair_col(vi+1, vi+2, "parent_same"),
                _mk_pair_col(vi+1, vi+2, "imgsim"),
                _mk_pair_col(vi+1, vi+2, "textsim"),
                _mk_pair_col(vi+1, vi+2, "ssim"),
            ]

        for c in cols:
            if c not in df.columns:
                df[c] = None

        df = df[cols]

        logging.info(f"[PERF] frame_materialization took {(time.perf_counter() - t_frame_start):.3f}s")

        return df

    except Exception as e:
        logging.error(f"Exception in build_n_version_dataframe: {e}")
        raise

def _copy_json_into_workspace(src: Path, dst_dir: Path) -> str:
    """
    Copy a JSON file into dst_dir and return the destination path string.
    If a file with the same name exists, add a numeric suffix to avoid overwrite.
    """
    _safe_makedirs(dst_dir)
    base = src.name
    dest = dst_dir / base
    if dest.exists():
        # add suffix
        stem = src.stem
        ext = src.suffix
        i = 1
        while True:
            cand = dst_dir / f"{stem}_{i}{ext}"
            if not cand.exists():
                dest = cand
                break
            i += 1
    try:
        shutil.copy2(src, dest)
        #logging.debug(f"[PREP] copied JSON {src} -> {dest}")
        return str(dest.resolve())
    except Exception as e:
        logging.warning(f"[PREP] failed to copy JSON {src} -> {dest} : {e}")
        return ""
    
### Handling ZIP, JSONs ####
def prepare_versions_and_images(
    inputs: List[str],
    workspace_root: str = None,
    key_column: dict = {}
) -> List[str]:
    """
    For each input path in `inputs`:
      - If zip:
          • extract images anywhere in zip → workspace_root/images (preserving filenames)
          • if structured.json exists → copy to workspace_root/structured_v{n}.json
          • else if any JSON exists inside zip → copy that JSON unchanged into workspace_root (preserve filename)
          • else if images exist → generate workspace_root/version_{n}.json (image-only listing)
      - If json:
          • copy json into workspace_root (preserve filename, avoid overwrite)

    Returns list of JSON paths (same length as inputs) to be used as VERSION_JSONS.
    """
    if workspace_root is None:
        raise ValueError("workspace_root must be provided (use processed_path)")
    #logging.debug(f"Versioning of image started")
    combined_dir = Path(workspace_root) / "images"
    #logging.debug(f"Combined path: {combined_dir}")
    out_dir = Path(workspace_root)
    #logging.debug(f"Output path: {out_dir}")
    _safe_makedirs(combined_dir)
    _safe_makedirs(out_dir)
    
    resulting_jsons: List[str] = []
    #logging.debug("Input processing started.")
    for idx, inp in enumerate(inputs):
        ver = idx + 1
        p = Path(inp)
        #logging.debug(f"[PREP] processing input #{ver} -> {inp}")

        if not p.exists():
            logging.warning(f"[PREP] input missing: {inp} -> skipping (empty slot)")
            resulting_jsons.append("")
            continue

        # ------------------------------------------
        # CASE 1: ZIP FILE
        # ------------------------------------------
        if p.suffix.lower() == ".zip":
            #logging.debug(f"File extension is zip")
            tmpdir = tempfile.mkdtemp(prefix=f"prep_v{ver}_")
            #logging.debug(f"tmpdir : {tmpdir}")
            #logging.debug(f"zip file path : {p}")
            #try:
            
            if not zipfile.is_zipfile(p):
                logging.warning(f"Not a valid ZIP archive: {p}")

            with zipfile.ZipFile(p, "r") as z:
                #logging.debug("Zip file extraction started.")
                z.extractall(tmpdir)
                #logging.debug("Zip file extraction completed.")
            tmpd = Path(tmpdir)
            #logging.debug(f"[PREP] extracted zip {p} -> {tmpdir}")

            # Store all image filenames discovered (preserving original names)
            found_image_filenames = []

            # Copy images from "images" directories first
            image_dirs = [d for d in tmpd.rglob("*") if d.is_dir() and d.name.lower() == "images"]
            images_found_count = 0

            if image_dirs:
                for img_dir in image_dirs:
                    for f in img_dir.rglob("*"):
                        if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
                            fname = f.name
                            found_image_filenames.append(fname)
                            dst = combined_dir / fname
                            if not dst.exists():
                                try:
                                    shutil.copy2(f, dst)
                                    images_found_count += 1
                                except Exception as e:
                                    logging.warning(f"[PREP] failed copying {f} -> {dst} : {e}")
                                    #raise Exception(f"[PREP] failed copying {f} -> {dst} : {e}")
            else:
                # fallback: copy any image anywhere
                for f in tmpd.rglob("*"):
                    if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
                        fname = f.name
                        found_image_filenames.append(fname)
                        dst = combined_dir / fname
                        if not dst.exists():
                            try:
                                shutil.copy2(f, dst)
                                images_found_count += 1
                            except Exception as e:
                                logging.warning(f"[PREP] failed copying {f} -> {dst} : {e}")
                                #raise Exception(f"[PREP] failed copying {f} -> {dst} : {e}")

            #logging.debug(f"[PREP] version {ver}: found {len(found_image_filenames)} images (copied {images_found_count})")

            # 1) Prefer explicit structured.json if present anywhere
            structured_found = None
            for cand in tmpd.rglob("structured.json"):
                if cand.is_file():
                    structured_found = cand
                    break

            if structured_found:
                # copy structured.json unchanged to structured_v{n}.json (avoid collision)
                out_struct = out_dir / f"structured_v{ver}.json"
                if out_struct.exists():
                    i = 1
                    while True:
                        candp = out_dir / f"structured_v{ver}_{i}.json"
                        if not candp.exists():
                            out_struct = candp
                            break
                        i += 1
                shutil.copy2(structured_found, out_struct)

                #===============Start: Removing the footer table if exists
                with open(structured_found, "r", encoding="utf-8") as f:
                        data = json.load(f)

                cleaned = []
                for entry in data:
                    if entry.get("type") == "table" and is_footer_table_html(entry.get("table_body", "")):
                        continue  # DROP footer table immediately
                    cleaned.append(entry)

                with open(out_struct, "w", encoding="utf-8") as f:
                    json.dump(cleaned, f, ensure_ascii=False, indent=2)
                #===============End: Removing the footer table if exists
                if key_column and p.suffix.lower() == ".zip":
                    filename = os.path.basename(p)
                    key_column_name = key_column.get(filename)
                    if key_column_name and key_column_name != "text":
                        #logging.debug(f"transforming json file {key_column_name}")
                        transform_json(out_struct, key_column_name)
                resulting_jsons.append(str(out_struct.resolve()))
                #logging.debug(f"[PREP] version {ver}: structured.json -> {out_struct}")

            else:
                # 2) If any JSON exists anywhere in the extracted tree, copy the shallowest JSON (preserve filename)
                json_candidates = [f for f in tmpd.rglob("*.json") if f.is_file()]
                if json_candidates:
                    # choose the shallowest (closest to tmpd root) to prefer top-level structured files like set_5.json
                    json_candidates.sort(key=lambda p: (len(p.relative_to(tmpd).parts), str(p).lower()))
                    chosen = json_candidates[0]
                    dst_json = out_dir / chosen.name
                    # avoid overwriting an existing file in workspace: pick an available name if needed
                    if dst_json.exists():
                        k = 1
                        while True:
                            candp = out_dir / f"{dst_json.stem}_{k}{dst_json.suffix}"
                            if not candp.exists():
                                dst_json = candp
                                break
                            k += 1
                    shutil.copy2(chosen, dst_json)
                    if key_column and p.suffix.lower() == ".zip":
                        filename = os.path.basename(p)
                        key_column_name = key_column.get(filename)
                        if key_column_name and key_column_name != "text":
                            #logging.debug(f"transforming json file {key_column_name}")
                            transform_json(dst_json, key_column_name)
                            resulting_jsons.append(str(dst_json.resolve()))
                    #logging.debug(f"[PREP] version {ver}: copied existing JSON {chosen} -> {dst_json}")

                else:
                    # 3) No JSON found; if images exist -> generate version_{n}.json listing image entries
                    unique_fnames = []
                    seen = set()
                    for fn in found_image_filenames:
                        if fn not in seen:
                            seen.add(fn)
                            unique_fnames.append(fn)

                    if unique_fnames:
                        gen = []
                        src_name = p.stem  # zip basename without .zip

                        for i, fname in enumerate(unique_fnames):
                            gen.append({
                                "type": "image",
                                "img_path": f"images/{fname}",
                                "page_idx": i,     # 0-based index as requested
                                "text": "",
                                "text_level": 0,  # requested default
                                "source_file": src_name,
                                "parent": ""
                            })

                        out_gen = out_dir / f"version_{ver}.json"
                        if out_gen.exists():
                            j = 1
                            while True:
                                candp = out_dir / f"version_{ver}_{j}.json"
                                if not candp.exists():
                                    out_gen = candp
                                    break
                                j += 1

                        with open(out_gen, "w", encoding="utf-8") as fo:
                            json.dump(gen, fo, ensure_ascii=False, indent=2)

                        #logging.debug(f"[PREP] version {ver}: generated {len(gen)} image entries -> {out_gen}")

                        resulting_jsons.append(str(out_gen.resolve()))
                    else:
                        # zip contains no images and no JSON
                        #logging.debug(f"[PREP] version {ver}: zip has no images and no JSON")
                        resulting_jsons.append("")
                        #raise Exception(f"[PREP] version {ver}: zip has no images and no JSON")

            # Best-effort cleanup for extracted zip working directory.
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

        # ------------------------------------------
        # CASE 2: JSON FILE (existing structured JSON)
        # ------------------------------------------
        elif p.suffix.lower() == ".json":
            #logging.debug(f"File extension is json")
            dst_json = _copy_json_into_workspace(p, out_dir)
            resulting_jsons.append(dst_json if dst_json else "")

        else:
            #logging.debug(f"File extension is unsupported")
            #logging.debug(f"[PREP] version {ver}: unsupported extension {p.suffix}")
            resulting_jsons.append("")

    return resulting_jsons

# =========================================================
# Main
# =========================================================

def create_mappings(revised_paths, type_filter, processed_path, key_column, task_id: str | None = None):
    # # Create mappings for revised paths
    # df = build_n_version_dataframe(revised_paths, type_filter=type_filter)
    # csv_name = f"{processed_path}/combined_mapping.csv"    
    # df.to_csv(csv_name, index=False, encoding="utf-8")
    # return None

    #configure_stdout_logging("INFO")
    # Prepare versions and combined images under cache
    #logging.debug(f"{revised_paths},{processed_path}")
    prepared = prepare_versions_and_images(revised_paths, workspace_root=processed_path,key_column=key_column)
    #logging.debug(f"Prepared JSONs per version ('' indicates missing structured.json for that slot): {prepared}")

    # Build list for pipeline: prefer prepared structured_vN (in workspace),
    # otherwise use copied JSON (if available) or leave "" to keep index
    jsons_for_pipeline = _resolve_jsons_for_pipeline(revised_paths, prepared, processed_path)

    usable_jsons = [p for p in jsons_for_pipeline if p]
    if len(usable_jsons) < 2:
        logging.error(f"Need at least two structured/json inputs for matching. Found: {len(usable_jsons)} usable JSON(s). Aborting.")
        raise SystemExit(1)

    combined_images = os.path.join(processed_path, "images")
    #logging.debug(f"Using combined images dir: {combined_images}")

    df = build_n_version_dataframe(
        jsons_for_pipeline,
        combined_images,
        type_filter=type_filter,
        task_id=task_id,
    )
    csv_name = os.path.join(processed_path, "combined_mapping.csv")
    df.to_csv(csv_name, index=False, encoding="utf-8")
    
    #logging.debug(f"combined_mapping.csv written: {df.shape} rows x {df.shape} cols")
    return None

def transform_json(path, key_to_change):
    # Load JSON data
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transformed_data = []

    for item in data:
        new_item = {}
        for key, value in item.items():
            
            if key==key_to_change:
                new_item["text"] = value
            else:
                new_item[key] = value
        transformed_data.append(new_item)

    # Save back to file
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, indent=4, ensure_ascii=False)


def _resolve_jsons_for_pipeline(revised_paths, prepared, processed_path):
    jsons_for_pipeline = []
    workspace_path = Path(processed_path)
    for original, prepared_path in zip(revised_paths, prepared):
        if prepared_path:
            jsons_for_pipeline.append(prepared_path)
            continue

        orig_p = Path(original)
        if orig_p.suffix.lower() == ".json":
            candidate = workspace_path / orig_p.name
            if candidate.exists():
                jsons_for_pipeline.append(str(candidate.resolve()))
            else:
                # fallback: use original path itself (keeps backward compatibility)
                jsons_for_pipeline.append(str(orig_p.resolve()))
        else:
            jsons_for_pipeline.append("")
    return jsons_for_pipeline

async def create_mappings_async(revised_paths, type_filter, processed_path, key_column, task_id: str | None = None):
    try:
        #logging.debug(f"{revised_paths},{processed_path}")
        loop = asyncio.get_running_loop()
        t_start = time.perf_counter()
        publish_interval_seconds = 10.0
        last_progress_publish_at = loop.time() - publish_interval_seconds
        pending_progress_event: Dict[str, Any] | None = None

        async def _publish_progress_event(event: Dict[str, Any]):
            if not task_id:
                return
            try:
                await create_queue_publish(f"delta_comparator_{task_id}", event)
            except Exception as publish_error:
                logging.warning(f"Dropped mapping progress event for task {task_id}: {publish_error}")

        async def _publish_pending_progress():
            nonlocal pending_progress_event, last_progress_publish_at
            if pending_progress_event is None:
                return
            now = loop.time()
            if (now - last_progress_publish_at) < publish_interval_seconds:
                return
            event = pending_progress_event
            pending_progress_event = None
            await _publish_progress_event(event)
            last_progress_publish_at = now

        async def _flush_pending_progress(force: bool = False):
            nonlocal pending_progress_event, last_progress_publish_at
            if pending_progress_event is None:
                return
            now = loop.time()
            if not force and (now - last_progress_publish_at) < publish_interval_seconds:
                return
            event = pending_progress_event
            pending_progress_event = None
            await _publish_progress_event(event)
            last_progress_publish_at = now

        # Offload zip/json preprocessing (disk + zip I/O) to a worker thread.
        t_prepare = time.perf_counter()
        prepared = await loop.run_in_executor(
            None,
            functools.partial(
                prepare_versions_and_images,
                revised_paths,
                workspace_root=processed_path,
                key_column=key_column,
            ),
        )
        logging.info(f"[PERF] prepare_versions_and_images took {(time.perf_counter() - t_prepare):.3f}s")
        #logging.debug(f"Prepared JSONs per version ('' indicates missing structured.json for that slot): {prepared}")

        jsons_for_pipeline = _resolve_jsons_for_pipeline(revised_paths, prepared, processed_path)

        usable_jsons = [p for p in jsons_for_pipeline if p]
        if len(usable_jsons) < 2:
            logging.error(f"Need at least two structured/json inputs for matching. Found: {len(usable_jsons)} usable JSON(s). Aborting.")
            raise Exception(f"Need at least two structured/json inputs for matching. Found: {len(usable_jsons)} usable JSON(s). Aborting.")

        combined_images = os.path.join(processed_path, "images")
        #logging.debug(f"Using combined images dir: {combined_images}")

        # Run build_n_version_dataframe in a thread pool (CPU bound, but pandas is not async)
        progress_queue = Queue()

        def _progress_sink(event: Dict[str, Any]):
            progress_queue.put(event)
            message = str(event.get("message") or "")
            if (
                "guarded" in message.lower()
                or "textvectorizer" in message.lower()
                or "image enrichment" in message.lower()
            ):
                logging.info(
                    f"kind={event.get('kind')} ratio={event.get('completed_ratio')} message={message} job_id={event.get('job_id')}"
                )

        try:
            t_match = time.perf_counter()
            future = loop.run_in_executor(
                None, build_n_version_dataframe, jsons_for_pipeline, combined_images, type_filter, task_id, _progress_sink
            )
            while not future.done():
                while True:
                    try:
                        event = progress_queue.get_nowait()
                    except Empty:
                        break
                    else:
                        pending_progress_event = event
                await _publish_pending_progress()
                await asyncio.sleep(0.1)

            while True:
                try:
                    event = progress_queue.get_nowait()
                except Empty:
                    break
                else:
                    pending_progress_event = event

            # Ensure the final buffered event is emitted even when interval has not elapsed yet.
            await _flush_pending_progress(force=True)

            df = await future
            logging.info(f"[PERF] build_n_version_dataframe took {(time.perf_counter() - t_match):.3f}s")
        except Exception as e:
            logging.error(f"Exception in build_n_version_dataframe: {e}")
            raise

        csv_name = os.path.join(processed_path, "combined_mapping.csv")
        # Offload CSV write (large DataFrames can block loop due to file I/O + serialization).
        t_csv = time.perf_counter()
        await loop.run_in_executor(
            None,
            functools.partial(df.to_csv, csv_name, index=False, encoding="utf-8"),
        )
        logging.info(f"[PERF] combined_mapping.csv write took {(time.perf_counter() - t_csv):.3f}s")
        #logging.debug(f"combined_mapping.csv written: {df.shape} rows x {df.shape} cols")
        logging.info(f"[PERF] create_mappings_async total took {(time.perf_counter() - t_start):.3f}s")
    except Exception as e:
        logging.error(f"Error in create_mappings_async: {e}")
        raise

# Performance tips:
# - Most of the heavy work is I/O and CPU-bound, so using run_in_executor is appropriate.
# - For further speedup, consider parallelizing independent I/O (e.g., processing multiple inputs in prepare_versions_and_images).
# - If build_n_version_dataframe is very slow, profile it for bottlenecks (e.g., pandas operations, matching loops).
# - If you have many large files, use a process pool for CPU-bound tasks (concurrent.futures.ProcessPoolExecutor).
# - Avoid unnecessary synchronous logging in hot loops.