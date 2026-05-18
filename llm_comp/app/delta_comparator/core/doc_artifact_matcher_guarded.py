# doc_artifact_matcher_guarded.py (fixed)
from __future__ import annotations
import os
import collections, re, math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from app.delta_comparator.utils.logger import log as logging

from app.delta_comparator.core.doc_artifact_matcher_plus import (
    # artifact building + corpus stopwords
    build_artifacts, build_indices, build_auto_stopwords,
    # shortlist + assignment util
    _shortlist, hungarian,
    # scoring
    score_table_robust as score_table,  # alias robust to legacy name
    score_image, score_text,
    # vectorizer
    TextVectorizer,
)

# ---------------------------
# Config
# ---------------------------

# Optional cross-kind fallback (table<->text)
ENABLE_CROSS_KIND_TABLE_TEXT = True
CROSS_KIND_TABLE_TEXT_THRESHOLD = 0.62  # conservative

# Near-match surfacing for TABLES (does not affect acceptance!)
NEAR_TABLE_MIN_SCORE = 0.45       # base floor for near-candidates
NEAR_TABLE_MAX_PER_SOURCE = 3     # top-k suggestions per source table
# A “structural/content” alternative when overall score is a hair low:
NEAR_HEADER_MIN = 0.25            # header_jacc or header_bigram_jacc
NEAR_COL_SEM_MIN = 0.30           # col_sem_top2
NEAR_ROWBAG_MIN = 0.20            # rowbag_jacc
NEAR_SEM_MIN = 0.40               # fallback semantic floor to pair with rowbag


def _batched(seq: List[Tuple[int, int]], batch_size: int):
    if batch_size <= 0:
        batch_size = len(seq) or 1
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def _build_parent_page_index(items: List[int], artifacts) -> Tuple[Dict[Any, List[int]], Dict[int, List[int]]]:
    by_parent: Dict[Any, List[int]] = collections.defaultdict(list)
    by_page: Dict[int, List[int]] = collections.defaultdict(list)
    for idx in items:
        a = artifacts[idx]
        by_parent[a.parent].append(idx)
        by_page[a.page_idx].append(idx)
    return dict(by_parent), dict(by_page)


def _cross_kind_candidates(
    src_indices: List[int],
    dst_by_parent: Dict[Any, List[int]],
    dst_by_page: Dict[int, List[int]],
    src_artifacts,
    page_window: int = 2,
) -> List[Tuple[int, int]]:
    """
    Build candidates matching original logic:
    same parent OR page distance <= page_window.
    Uses indexed lookups to avoid O(n^2) scans.
    """
    out: List[Tuple[int, int]] = []
    for i in src_indices:
        a = src_artifacts[i]
        candidates = set(dst_by_parent.get(a.parent, []))
        for p in range(a.page_idx - page_window, a.page_idx + page_window + 1):
            candidates.update(dst_by_page.get(p, []))
        if not candidates:
            continue
        # Stable order by destination index preserves deterministic downstream sorting.
        for j in sorted(candidates):
            out.append((i, j))
    return out

# ---- helper: safe signal packing (prevents float('base') errors) ----
def _pack_sigs(sigs: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (sigs or {}).items():
        if isinstance(v, (int, float)):
            if math.isfinite(float(v)):
                out[k] = round(float(v), 4)
            else:
                out[k] = float(v)
        else:
            # keep non-numeric (e.g., "path": "base"/"numeric") as-is
            out[k] = v
    return out

def _pairs_for_kind(A, B, kind, tv, k=24):
    indexB = build_indices(B, tv=tv, kind=kind)
    pairs = []
    k_local = 48 if kind == "text" else (32 if kind == "table" else k)
    for i, a in enumerate(A):
        if a.type != kind:
            continue
        cand = _shortlist(a, B, indexB, kind, k=k_local, tv=tv, ann_top_k=max(32, k_local))

        # also ensure any exact normalized-text matches are present
        try:
            exacts = indexB.get("text_exact", {}).get(a.text_clean, set())
            cand = list(dict.fromkeys(list(cand) + list(exacts)))
        except Exception:
            pass

        for j in cand:
            if B[j].type == kind:
                pairs.append((i, j))
    return pairs

def _collect_all_scores(A, B, kind, tv, pairs, progress_callback=None):
    rows = sorted(set(i for i, _ in pairs))
    cols = sorted(set(j for _, j in pairs))
    rix = {r: k for k, r in enumerate(rows)}
    cix = {c: k for k, c in enumerate(cols)}
    mat = np.ones((len(rows), len(cols)), dtype=np.float32)
    score_map = {}
    signal_map = {}
    total_pairs = len(pairs)
    next_checkpoint = 10
    for pair_index, (i, j) in enumerate(pairs, start=1):
        if kind == "table":
            s, sigs = score_table(A[i], B[j], tv)
        elif kind == "image":
            s, sigs = score_image(A[i], B[j], tv)
        else:
            s, sigs = score_text(A[i], B[j], tv)
        mat[rix[i]][cix[j]] = 1.0 - s
        score_map[(i, j)] = s
        signal_map[(i, j)] = sigs
        if progress_callback and total_pairs:
            percent = int(pair_index / total_pairs * 100)
            if pair_index == total_pairs or percent >= next_checkpoint:
                progress_callback(pair_index, total_pairs)
                while next_checkpoint <= percent:
                    next_checkpoint += 10
    return mat, rows, cols, score_map, signal_map


def _emit_progress(
    progress_sink: Optional[Callable[[Dict[str, Any]], None]],
    task_id: str | None,
    message: str,
    completed_ratio: int,
    kind: str | None = None,
) -> None:
    if not task_id or progress_sink is None:
        return
    progress_sink({
        "type": "stream",
        "task_id": task_id,
        "kind": kind,
        "completed_ratio": completed_ratio,
        "message": message,
    })

def _best_two(scores, axis='row'):
    by = collections.defaultdict(list)
    for (i, j), s in scores.items():
        key = i if axis == 'row' else j
        val = (j if axis == 'row' else i, s)
        by[key].append(val)
    out = {}
    for k, arr in by.items():
        arr.sort(key=lambda x: -x[1])
        out[k] = arr[:2] + [(-1, 0.0)] * (2 - len(arr))
    return out

def _apply_policy(A, B, kind, assign, rows, cols, score_map, signal_map, threshold, policy):
    min_margin     = policy.get("min_margin", 0.08)
    parent_penalty = policy.get("parent_penalty", 0.05)
    page_soft      = policy.get("page_distance_soft", 40)
    page_penalty   = policy.get("page_penalty", 0.05)
    min_signals    = policy.get("min_signals", {})

    def _passes_signal_floors(kind, sigs, floors):
        floors_k = floors.get(kind, {})
        if not floors_k:
            return True
        for name, floor in floors_k.items():
            if name == "chars_or_sem":
                if max(sigs.get("chars", 0.0), sigs.get("semantic", 0.0)) < floor:
                    return False
            elif name == "containment":
                if max(sigs.get("contain_tok", 0.0), sigs.get("contain_char", 0.0)) < floor:
                    return False
            else:
                if sigs.get(name, 0.0) < floor:
                    return False
        return True

    def _len_ratio(ta: int, tb: int) -> float:
        if ta == 0 or tb == 0: return 99.0
        big, small = (ta, tb) if ta >= tb else (tb, ta)
        return big / max(1, small)

    best_row = _best_two(score_map, axis='row')
    best_col = _best_two(score_map, axis='col')

    accepted = []
    used_rows = set()
    used_cols = set()

    for (ri, cj) in assign:
        i = rows[ri]; j = cols[cj]
        s = score_map.get((i, j), 0.0)
        sigs = signal_map.get((i, j), {})

        thr = threshold
        same_parent = (A[i].parent == B[j].parent)
        page_gap = abs(A[i].page_idx - B[j].page_idx)
        if not same_parent:
            thr += parent_penalty
        if page_gap > page_soft:
            thr += page_penalty

        if not _passes_signal_floors(kind, sigs, min_signals):
            continue

        if kind == "text":
            tokA = len(A[i].tokens); tokB = len(B[j].tokens)
            lr = _len_ratio(tokA, tokB)
            a_head = getattr(A[i], "is_heading", False)
            b_head = getattr(B[j], "is_heading", False)
            heading_mismatch = (a_head != b_head)

            contain_char   = sigs.get("contain_char", 0.0)
            contain_tok    = sigs.get("contain_tok", 0.0)
            contain_labels = sigs.get("contain_labels", 0.0)
            sg_cover       = sigs.get("shingle_cover", 0.0)
            sem            = sigs.get("semantic", 0.0)
            sem_labels     = sigs.get("sem_labels", 0.0)
            anchor_jacc    = sigs.get("anchor_jacc", 0.0)
            anchor_cover   = sigs.get("anchor_cover", 0.0)

            if min(tokA, tokB) <= 4:
                if not ((contain_char >= 0.98 or contain_tok >= 0.80) and (same_parent or page_gap <= 2)):
                    continue

            if (lr > 2.5) and (contain_char < 0.95) and (sg_cover < 0.06) and (max(sem, sem_labels) < 0.82):
                continue

            if heading_mismatch and (contain_char < 0.95) and (contain_tok < 0.75) and (sg_cover < 0.06) and (max(sem, sem_labels) < 0.82):
                continue

            near_para = (sem_labels >= 0.90 and anchor_jacc >= 0.50) or (contain_char >= 0.985)
            if near_para and (same_parent or page_gap <= 3):
                if s >= thr:
                    accepted.append((i, j, s, sigs))
                    used_rows.add(i); used_cols.add(j)
                continue

            if max(tokA, tokB) >= 25:
                if max(anchor_cover, anchor_jacc, sg_cover) < 0.05:
                    continue
        # if kind == "table":
        #     header_j = sigs.get("header_jacc", 0.0)
        #     rowbag = sigs.get("rowbag_jacc", 0.0)
        #     col_sem = sigs.get("col_sem_top2", 0.0)
        #     sem = sigs.get("semantic", 0.0)

        #     #  HARD REJECTION: structurally different tables
        #     if header_j < 0.2 and rowbag < 0.15:
        #         continue

        #     # HARD REJECTION: weak semantic + weak structure
        #     if sem < 0.5 and col_sem < 0.3:
        #         continue

        #     # HARD REJECTION: row mismatch (VERY IMPORTANT)
        #     ra = len(A[i].grid_cells)
        #     rb = len(B[j].grid_cells)
        #     if min(ra, rb) > 0:
        #         row_ratio = max(ra, rb) / min(ra, rb)
        #         if row_ratio > 3:
        #             continue
        
        if kind == "table":
            header_j = sigs.get("header_jacc", 0.0)
            rowbag = sigs.get("rowbag_jacc", 0.0)
            col_sem = sigs.get("col_sem_top2", 0.0)
            sem = sigs.get("semantic", 0.0)
            #2024-03-24: NEW: Row overlap signal (important for extraction drift cases where row count can change drastically)
            row_overlap = sigs.get("row_overlap", 0.0)
            #  Semantic override (BERT should dominate)
            strong_semantic = sem >= 0.72 #0.65
            reg_jacc = sigs.get("reg_jacc", 0.0)
            # Relax structure rejection
            if not strong_semantic:
                if header_j < 0.1 and rowbag < 0.1:
                    continue
            
            #25 03 2026-- NEW: REG_JACC STRUCTURAL GUARD (prevents semantic leakage in extreme extraction drift cases)
            if reg_jacc < 0.15:
                # Only allow if VERY strong structural match
                if not (
                    row_overlap >= 0.45 and
                    rowbag >= 0.30 and
                    sem >= 0.70
                ):
                    continue

            # Relax semantic gating
            if not strong_semantic:
                if sem < 0.4 and col_sem < 0.25:
                    continue
            #2024-03-24: NEW: Row overlap gating (important for extraction drift cases where row count can change drastically)
            # if row_overlap < 0.25 and sem < 0.65:
            #     continue
            #Latest
            #Latest: night STRICTER: require some structure unless very strong semantic
            if row_overlap < 0.30:   #0.25
                # if not (
                #     sem >= 0.75 and
                #     (col_sem >= 0.5 or rowbag >= 0.25)
                # ):
                if not (
                    sem >= 0.78 and
                    (col_sem >= 0.55 or rowbag >= 0.30)
                ):
                    continue
            # Relax row mismatch
            ra = len(A[i].grid_cells)
            rb = len(B[j].grid_cells)
            if min(ra, rb) > 0:
                row_ratio = max(ra, rb) / min(ra, rb)
                if row_ratio > 5 and not strong_semantic:
                    continue
        # Mutual-best + margin
        br = best_row.get(i, [(-1, 0.0), (-1, 0.0)])
        bc = best_col.get(j, [(-1, 0.0), (-1, 0.0)])
        r_best_col, r_best_score = br[0]
        c_best_row, c_best_score = bc[0]

        skip_mutual = False
        if kind == "text":
            if (min(len(A[i].tokens), len(B[j].tokens)) <= 6 and sigs.get("contain_char", 0.0) >= 0.975 and (same_parent or page_gap <= 2)):
                skip_mutual = True
            elif max(sigs.get("contain_char", 0.0), sigs.get("chars", 0.0)) >= 0.98:
                skip_mutual = True

        if not skip_mutual:
            if not (r_best_col == j and c_best_row == i):
                if sigs.get("semantic", 0.0) < 0.7:
                    continue
            r_second = br[1][1] if len(br) > 1 else 0.0
            c_second = bc[1][1] if len(bc) > 1 else 0.0
            eff_min_margin = min_margin * (0.5 if s >= 0.78 else 1.0)
            if (r_best_score - r_second) < eff_min_margin:
                continue
            if (c_best_score - c_second) < eff_min_margin:
                continue

        if s >= thr:
            accepted.append((i, j, s, sigs))
            used_rows.add(i)
            used_cols.add(j)

    A_ids = [idx for idx, a in enumerate(A) if a.type == kind]
    B_ids = [idx for idx, b in enumerate(B) if b.type == kind]
    unmatched = {
        "source":  [A[i].id for i in A_ids if i not in used_rows],
        "revision":[B[j].id for j in B_ids if j not in used_cols]
    }
    return accepted, unmatched

# ---------------------------
# Build near table matches (suggestions)
# ---------------------------

def _build_near_table_matches(
    kind: str,
    A, B,
    pairs: List[Tuple[int,int]],
    score_map: Dict[Tuple[int,int], float],
    signal_map: Dict[Tuple[int,int], Dict[str,float]],
    accepted_pairs: List[Tuple[int,int]],
    max_per_source: int = NEAR_TABLE_MAX_PER_SOURCE,
) -> List[Dict[str, Any]]:
    if kind != "table":
        return []

    accepted_set = set(accepted_pairs)
    by_source: Dict[int, List[Tuple[int, float]]] = collections.defaultdict(list)
    for (i, j), s in score_map.items():
        if (i, j) in accepted_set:
            continue
        by_source[i].append((j, s))

    out = []
    for i, arr in by_source.items():
        arr.sort(key=lambda x: -x[1])
        taken = 0
        rank = 1
        for j, s in arr:
            if taken >= max_per_source:
                break
            sigs = signal_map.get((i, j), {})

            ok_score = (s >= NEAR_TABLE_MIN_SCORE)

            hdr = max(sigs.get("header_jacc", 0.0), sigs.get("header_bigram_jacc", 0.0))
            col = sigs.get("col_sem_top2", 0.0)
            rowb = sigs.get("rowbag_jacc", 0.0)
            semv = sigs.get("semantic", 0.0)

            ok_alt = (hdr >= NEAR_HEADER_MIN and col >= NEAR_COL_SEM_MIN) \
                     or (rowb >= NEAR_ROWBAG_MIN and semv >= NEAR_SEM_MIN)

            if not (ok_score or ok_alt):
                continue

            ai, bj = A[i], B[j]
            out.append({
                "source_id": ai.id,
                "revision_id": bj.id,
                "score": round(float(s), 4),
                "signals": _pack_sigs(sigs),
                "source_meta": {"page_idx": ai.page_idx, "parent": ai.parent, "order": ai.order},
                "revision_meta": {"page_idx": bj.page_idx, "parent": bj.parent, "order": bj.order},
                "source_text": ai.text_orig,
                "revision_text": bj.text_orig,
                "source_img_path": ai.img_path,
                "revision_img_path": bj.img_path,
                "source_table_body": ai.html if ai.type == "table" else None,
                "revision_table_body": bj.html if bj.type == "table" else None,
                "near_reason": "score>=min OR header/col/rowbag strong",
                "rank_for_source": rank,
            })
            taken += 1
            rank += 1

    out.sort(key=lambda x: (-x["score"], x["rank_for_source"]))
    return out

# ---------------------------
# Main (artifacts already built)
# ---------------------------

def match_one_revision_guarded(A, B, thresholds, policy, task_id: str | None = None, progress_start: int = 30, progress_end: int = 48, progress_sink: Optional[Callable[[Dict[str, Any]], None]] = None):
    """
    Use this when you already built artifacts A & B (possibly with a corpus stopword set).
    """

    tv = TextVectorizer()
    out = {}
    unmatched_out = {}
    near_out = {"table": [], "image": [], "text": []}  # new: suggestions

    # Primary, same-kind matching
    kinds = ("table", "text", "image")
    total_kinds = len(kinds)
    for kind_index, kind in enumerate(kinds):
        #logging.debug(f"Starting matching for kind: {kind}")
        kind_start = progress_start + int((progress_end - progress_start) * kind_index / total_kinds)
        kind_end = progress_start + int((progress_end - progress_start) * (kind_index + 1) / total_kinds)
        display_kind = "tables" if kind == "table" else ("text sections" if kind == "text" else "images")
        emit_kind_progress = kind != "image"
        if emit_kind_progress:
            _emit_progress(progress_sink, task_id, f"Reviewing {display_kind}", kind_start, kind)
        pairs = _pairs_for_kind(A, B, kind, tv, k=24)
        #logging.debug(f"Found {len(pairs)} candidate pairs for kind: {kind}")
        if not pairs:
            out[kind] = []
            unmatched_out[kind] = {
                "source":  [a.id for a in A if a.type == kind],
                "revision":[b.id for b in B if b.type == kind]
            }
            if emit_kind_progress:
                _emit_progress(progress_sink, task_id, f"Finished reviewing {display_kind}", kind_end, kind)
            #logging.debug(f"No pairs to process for kind: {kind}")
            continue

        def progress_callback(processed_pairs: int, total_pairs: int):
            if not emit_kind_progress:
                return
            completed_ratio = kind_end if processed_pairs >= total_pairs else kind_start + int((kind_end - kind_start) * processed_pairs / total_pairs)
            _emit_progress(
                progress_sink,
                task_id,
                f"Reviewing {display_kind}: {processed_pairs}/{total_pairs} checked",
                completed_ratio,
                kind,
            )

        mat, rows, cols, score_map, signal_map = _collect_all_scores(A, B, kind, tv, pairs, progress_callback=progress_callback)
        ##logging.debug(f"Collected all scores for kind: {kind}")
        assign = hungarian(mat)
        #logging.debug(f"Hungarian assignment done for kind: {kind}")
        accepted, unmatched = _apply_policy(
            A, B, kind, assign, rows, cols, score_map, signal_map,
            thresholds.get(kind, 0.6), policy
        )
        #logging.debug(f"Applied policy for kind: {kind}, accepted: {len(accepted)}, unmatched: {unmatched}")
        unmatched_out[kind] = unmatched

        # pack accepted
        pk = []
        accepted_pairs = []
        for (ai, bj, score, sigs) in sorted(accepted, key=lambda x: -x[2]):
            accepted_pairs.append((ai, bj))
            pk.append({
                "source_id": A[ai].id,
                "revision_id": B[bj].id,
                "score": round(float(score), 4),
                "signals": _pack_sigs(sigs),
                "source_meta": {"page_idx": A[ai].page_idx, "parent": A[ai].parent, "order": A[ai].order},
                "revision_meta": {"page_idx": B[bj].page_idx, "parent": B[bj].parent, "order": B[bj].order},
                "source_text": A[ai].text_orig,
                "revision_text": B[bj].text_orig,
                "source_img_path": A[ai].img_path,
                "revision_img_path": B[bj].img_path,
                "source_table_body": A[ai].html if A[ai].type == "table" else None,
                "revision_table_body": B[bj].html if B[bj].type == "table" else None,
            })
        out[kind] = pk
        ##logging.debug(f"Packed {len(pk)} accepted matches for kind: {kind}")
        if emit_kind_progress:
            _emit_progress(progress_sink, task_id, f"Finished reviewing {display_kind}", kind_end, kind)

        # build near suggestions for tables
        if kind == "table":
            near_out["table"] = _build_near_table_matches(
                kind, A, B, pairs, score_map, signal_map, accepted_pairs,
                max_per_source=NEAR_TABLE_MAX_PER_SOURCE
            )
            #logging.debug(f"Built {len(near_out['table'])} near table matches")
        ### Clear cache
        #logging.debug(f"[Cache] Clearing after {kind}. Size: {len(tv.cache)}")
        tv.clear_cache()
    # Cross-kind fallback: Table ↔ Text (for extraction drift)
    if ENABLE_CROSS_KIND_TABLE_TEXT:
        text_stage_end = progress_start + int((progress_end - progress_start) * 2 / 3)
        cross_stage_start = max(progress_start, text_stage_end)
        cross_stage_end = max(cross_stage_start + 1, progress_end - 1)
        #logging.debug("Starting cross-kind table-text fallback matching")
        tables_A = [i for i, a in enumerate(A) if a.type == "table"]
        texts_B  = [j for j, b in enumerate(B) if b.type == "text"]
        tables_B = [j for j, b in enumerate(B) if b.type == "table"]
        texts_A  = [i for i, a in enumerate(A) if a.type == "text"]

        texts_B_by_parent, texts_B_by_page = _build_parent_page_index(texts_B, B)
        tables_B_by_parent, tables_B_by_page = _build_parent_page_index(tables_B, B)

        # Flatten nested scans via indexed candidate generation (same eligibility logic).
        cross_pairs_1 = _cross_kind_candidates(tables_A, texts_B_by_parent, texts_B_by_page, A, page_window=2)
        cross_pairs_2 = _cross_kind_candidates(texts_A, tables_B_by_parent, tables_B_by_page, A, page_window=2)
        total_cross_pairs = len(cross_pairs_1) + len(cross_pairs_2)
        processed_cross_pairs = 0
        next_cross_checkpoint = 10

        if total_cross_pairs > 0:
            _emit_progress(
                progress_sink,
                task_id,
                "Running final consistency checks",
                cross_stage_start,
                "text",
            )

        #logging.debug(f"Cross-kind pairs: table->text: {len(cross_pairs_1)}, text->table: {len(cross_pairs_2)}")

        def _score_cross(i, j, a_is_table: bool):
            from_text = (A[i].table_flat_text if a_is_table else A[i].text_clean)
            to_text   = (B[j].table_flat_text if not a_is_table else B[j].text_clean)
            s, sigs = score_text(
                type("Tmp", (), {"text_clean": from_text, "text_semantic": re.sub(r'\b\d+\b', '<num>', from_text),
                                 "simhash_text": A[i].simhash_text, "tokens": A[i].tokens,
                                 "anchor_tokens": A[i].anchor_tokens})(),
                type("Tmp", (), {"text_clean": to_text, "text_semantic": re.sub(r'\b\d+\b', '<num>', to_text),
                                 "simhash_text": B[j].simhash_text, "tokens": B[j].tokens,
                                 "anchor_tokens": B[j].anchor_tokens})(),
                tv
            )
            return s, sigs

        accepted_cross = []

        # Chunk scoring to reduce long single-loop stalls and keep memory steady.
        cross_batch = max(128, int(os.getenv("CROSS_KIND_SCORE_BATCH_SIZE", "512")))
        for batch in _batched(cross_pairs_1, cross_batch):
            for (i, j) in batch:
                s, sigs = _score_cross(i, j, a_is_table=True)
                if s >= CROSS_KIND_TABLE_TEXT_THRESHOLD:
                    accepted_cross.append(("table->text", i, j, s, sigs))
            if total_cross_pairs > 0:
                processed_cross_pairs = min(total_cross_pairs, processed_cross_pairs + len(batch))
                percent_done = int((processed_cross_pairs / total_cross_pairs) * 100)
                if processed_cross_pairs == total_cross_pairs or percent_done >= next_cross_checkpoint:
                    cross_ratio = min(
                        cross_stage_end,
                        cross_stage_start + int((cross_stage_end - cross_stage_start) * processed_cross_pairs / max(1, total_cross_pairs)),
                    )
                    _emit_progress(
                        progress_sink,
                        task_id,
                        f"Running final consistency checks: {processed_cross_pairs}/{total_cross_pairs} reviewed",
                        cross_ratio,
                        "text",
                    )
                    while next_cross_checkpoint <= percent_done:
                        next_cross_checkpoint += 10
        for batch in _batched(cross_pairs_2, cross_batch):
            for (i, j) in batch:
                s, sigs = _score_cross(i, j, a_is_table=False)
                if s >= CROSS_KIND_TABLE_TEXT_THRESHOLD:
                    accepted_cross.append(("text->table", i, j, s, sigs))
            if total_cross_pairs > 0:
                processed_cross_pairs = min(total_cross_pairs, processed_cross_pairs + len(batch))
                percent_done = int((processed_cross_pairs / total_cross_pairs) * 100)
                if processed_cross_pairs == total_cross_pairs or percent_done >= next_cross_checkpoint:
                    cross_ratio = min(
                        cross_stage_end,
                        cross_stage_start + int((cross_stage_end - cross_stage_start) * processed_cross_pairs / max(1, total_cross_pairs)),
                    )
                    _emit_progress(
                        progress_sink,
                        task_id,
                        f"Running final consistency checks: {processed_cross_pairs}/{total_cross_pairs} reviewed",
                        cross_ratio,
                        "text",
                    )
                    while next_cross_checkpoint <= percent_done:
                        next_cross_checkpoint += 10

        #logging.debug(f"Accepted cross-kind matches: {len(accepted_cross)}")

        matched_A = set([m["source_id"] for m in out.get("table", [])] + [m["source_id"] for m in out.get("text", [])])
        matched_B = set([m["revision_id"] for m in out.get("table", [])] + [m["revision_id"] for m in out.get("text", [])])

        for tag, ai, bj, score, sigs in sorted(accepted_cross, key=lambda x: -x[3]):
            if (A[ai].id in matched_A) or (B[bj].id in matched_B):
                continue
            if A[ai].type == "table":
                out["table"].append({
                    "source_id": A[ai].id,
                    "revision_id": B[bj].id,
                    "score": round(float(score), 4),
                    "signals": _pack_sigs(sigs),
                    "source_meta": {"page_idx": A[ai].page_idx, "parent": A[ai].parent, "order": A[ai].order},
                    "revision_meta": {"page_idx": B[bj].page_idx, "parent": B[bj].parent, "order": B[bj].order},
                    "source_text": A[ai].text_orig,
                    "revision_text": B[bj].text_orig,
                    "source_img_path": A[ai].img_path,
                    "revision_img_path": B[bj].img_path,
                    "source_table_body": A[ai].html if A[ai].type == "table" else None,
                    "revision_table_body": B[bj].html if B[bj].type == "table" else None,
                })
                matched_A.add(A[ai].id); matched_B.add(B[bj].id)
            elif B[bj].type == "table":
                out["table"].append({
                    "source_id": A[ai].id,
                    "revision_id": B[bj].id,
                    "score": round(float(score), 4),
                    "signals": _pack_sigs(sigs),
                    "source_meta": {"page_idx": A[ai].page_idx, "parent": A[ai].parent, "order": A[ai].order},
                    "revision_meta": {"page_idx": B[bj].page_idx, "parent": B[bj].parent, "order": B[bj].order},
                    "source_text": A[ai].text_orig,
                    "revision_text": B[bj].text_orig,
                    "source_img_path": A[ai].img_path,
                    "revision_img_path": B[bj].img_path,
                    "source_table_body": A[ai].html if A[ai].type == "table" else None,
                    "revision_table_body": B[bj].html if B[bj].type == "table" else None,
                })
                matched_A.add(A[ai].id); matched_B.add(B[bj].id)

        if total_cross_pairs > 0:
            _emit_progress(
                progress_sink,
                task_id,
                "Final consistency checks completed",
                cross_stage_end,
                "text",
            )

    summary = {
        "source_counts":   {k: sum(1 for a in A if a.type == k) for k in ("table","image","text")},
        "revision_counts": {k: sum(1 for b in B if b.type == k) for k in ("table","image","text")},
        "matched":         {k: len(out.get(k, [])) for k in ("table","image","text")},
        "near_table":      len(near_out.get("table", [])),
    }
    #logging.debug(f"Matching summary: {summary}")
    return {"summary": summary, "matches": out, "unmatched": unmatched_out, "near_matches": near_out}

# ---------------------------
# Convenience: raw JSON → artifacts → guarded
# ---------------------------

def match_jsons_guarded(A_raw: List[Dict[str,Any]],
                        B_raw: List[Dict[str,Any]],
                        image_dir: str,
                        thresholds: Dict[str, float],
                        policy: Dict[str, Any]):
    """
    Convenience entry point when you have raw JSON entries (dicts) for A and B.
    """
    A_texts = [e.get("text", "") for e in A_raw if e.get("type") in ("table","image","text")]
    B_texts = [e.get("text", "") for e in B_raw if e.get("type") in ("table","image","text")]
    STOPSET = build_auto_stopwords(A_texts + B_texts, df_cutoff=0.60)

    A = build_artifacts(A_raw, image_dir=image_dir, stopset=STOPSET)
    B = build_artifacts(B_raw, image_dir=image_dir, stopset=STOPSET)

    res = match_one_revision_guarded(A, B, thresholds=thresholds, policy=policy)
    return res, A, B, STOPSET
