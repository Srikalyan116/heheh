# match_pipeline.py
from __future__ import annotations
from typing import Dict, Any, List, Iterable, Set, Callable, Optional, Awaitable, cast
from app.delta_comparator.utils.logger import log as logging
import asyncio
import time
from app.delta_comparator.core.doc_artifact_matcher_plus import build_artifacts, build_auto_stopwords
from app.delta_comparator.core.doc_artifact_matcher_guarded import match_one_revision_guarded
from app.delta_comparator.core.image_prepass_matcher import enrich_images_with_ocr_and_summary, match_images_prepass

# ----------------------------
# helpers
# ----------------------------
_ALLOWED = {"image", "table", "text"}


def _emit_progress(
    progress_sink: Optional[Callable[[Dict[str, Any]], None]],
    task_id: str | None,
    message: str,
    completed_ratio: int,
    kind: str | None = None,
) -> None:
    logging.debug(f"Progress update: {message} (ratio={completed_ratio}%)")
    if not task_id or progress_sink is None:
        logging.debug(f"Progress sink not available, skipping progress update for message: {message} and task_id: {task_id}")
        return
    progress_sink({
        "type": "stream",
        "task_id": task_id,
        "kind": kind,
        "completed_ratio": completed_ratio,
        "message": message,
    })

def _normalize_kind_filter(kinds: Any) -> Set[str]:
    if kinds is None:
        return set(_ALLOWED)
    if isinstance(kinds, str):
        s = kinds.strip().lower()
        if s in ("all", "*"):
            return set(_ALLOWED)
        return {s} & _ALLOWED
    try:
        return set(str(k).strip().lower() for k in kinds) & _ALLOWED
    except Exception:
        return set(_ALLOWED)

def _filter_raw_by_kind(raw: List[Dict[str, Any]], kinds: Set[str]) -> List[Dict[str, Any]]:
    if not kinds or kinds == _ALLOWED:
        return raw
    out = []
    for e in raw:
        t = str(e.get("type","text")).strip().lower()
        if t in kinds:
            out.append(e)
    return out


def match_revision_image_first(
    A_raw: List[Dict[str,Any]],
    B_raw: List[Dict[str,Any]],
    image_dir: str,
    thresholds: Dict[str, float],
    policy: Dict[str, Any],
    image_threshold: float = 0.55,
    shortlist_k: int = 24,
    run_kinds: Any = None,
    task_id: str | None = None,
    progress_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:

    logging.info("Starting match_revision_image_first pipeline")
    stage_t0 = time.perf_counter()

    # --- Normalize kinds ---
    kinds = _normalize_kind_filter(run_kinds) or set(_ALLOWED)
    #logging.debug(f"Kind filter active: {sorted(kinds)}")

    # --- Filter raw inputs ---
    A_raw_f = _filter_raw_by_kind(A_raw, kinds)
    B_raw_f = _filter_raw_by_kind(B_raw, kinds)

    #logging.debug(f"A_raw_f size: {len(A_raw_f)}")
    #logging.debug(f"B_raw_f size: {len(B_raw_f)}")

    # --- Build stopwords (optimized input extraction) ---
    A_texts = [e.get("text", "") for e in A_raw_f if e.get("text")]
    B_texts = [e.get("text", "") for e in B_raw_f if e.get("text")]

    STOPSET = build_auto_stopwords(A_texts + B_texts, df_cutoff=0.60)
    #logging.debug(f"Auto-stopwords built: {len(STOPSET)} tokens")

    # --- Build artifacts ---
    A = build_artifacts(A_raw_f, image_dir=image_dir, stopset=STOPSET)
    B = build_artifacts(B_raw_f, image_dir=image_dir, stopset=STOPSET)

    #logging.debug(f"Artifacts built -> source: {len(A)}, revision: {len(B)}")

    # --- IMAGE ENRICHMENT (PARALLEL FIX) ---
    if "image" in kinds:
        enrich_t0 = time.perf_counter()
        logging.info("Starting parallel image enrichment")
        source_total_images = sum(1 for a in A if a.type == "image")
        revised_total_images = sum(1 for b in B if b.type == "image")
        total_images = source_total_images + revised_total_images
        progress_state = {
            "source_processed": 0,
            "revised_processed": 0,
        }

        def _extract_processed_count(message: str) -> Optional[int]:
            if not message:
                return None
            for token in str(message).replace(":", " ").split():
                if "/" not in token:
                    continue
                left, _, right = token.partition("/")
                if left.isdigit() and right.isdigit():
                    return int(left)
            return None

        def _wrapped_progress_callback(event: Dict[str, Any]) -> None:
            message = str(event.get("message") or "")
            processed = _extract_processed_count(message)
            if processed is not None:
                lower_msg = message.lower()
                if "source image enrichment" in lower_msg:
                    progress_state["source_processed"] = min(processed, source_total_images)
                elif "revised image enrichment" in lower_msg:
                    progress_state["revised_processed"] = min(processed, revised_total_images)
            # Intentionally do not forward raw per-image callbacks.
            # We emit throttled image-enrichment progress on heartbeat interval.

        async def _run_image_enrichment():
            heartbeat_interval_seconds = 10.0
            heartbeat_stop = asyncio.Event()

            async def _heartbeat():
                while not heartbeat_stop.is_set():
                    await asyncio.sleep(heartbeat_interval_seconds)
                    if heartbeat_stop.is_set():
                        break
                    processed_images = (
                        min(progress_state["source_processed"], source_total_images)
                        + min(progress_state["revised_processed"], revised_total_images)
                    )
                    ratio = 36 if total_images <= 0 else 12 + int((36 - 12) * processed_images / max(1, total_images))
                    _emit_progress(
                        progress_sink,
                        task_id,
                        f"Preparing images: {processed_images}/{total_images} processed",
                        ratio,
                        "image",
                    )

            heartbeat_task = asyncio.create_task(_heartbeat())
            try:
                src_enrich = cast(
                    Awaitable[Any],
                    enrich_images_with_ocr_and_summary(
                        A,
                        image_root=image_dir,
                        debug=False,
                        side="v1",
                        concurrency=6,
                        task_id=task_id,
                        progress_start=12,
                        progress_end=24,
                        stage_name="source image enrichment",
                        progress_callback=_wrapped_progress_callback,
                    ),
                )
                rev_enrich = cast(
                    Awaitable[Any],
                    enrich_images_with_ocr_and_summary(
                        B,
                        image_root=image_dir,
                        debug=False,
                        side="v2",
                        concurrency=6,
                        task_id=task_id,
                        progress_start=24,
                        progress_end=36,
                        stage_name="revised image enrichment",
                        progress_callback=_wrapped_progress_callback,
                    ),
                )
                await asyncio.gather(
                    src_enrich,
                    rev_enrich,
                )
            finally:
                heartbeat_stop.set()
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)

        #logging.debug("Running parallel image enrichment")
        _emit_progress(progress_sink, task_id, "Preparing images for comparison", 12, "image")

        asyncio.run(_run_image_enrichment())
        logging.info(f"Image enrichment completed in {time.perf_counter() - enrich_t0:.2f}s")
        _emit_progress(progress_sink, task_id, "Image preparation completed", 36, "image")

    # --- IMAGE PREPASS ---
    image_rows, image_unmatched = [], {"source": [], "revision": []}

    if "image" in kinds:
        _emit_progress(progress_sink, task_id, "Comparing images", 36, "image")
        prepass_t0 = time.perf_counter()
        logging.info("Running image prepass matching")

        image_rows, image_unmatched = match_images_prepass(
            A, B, threshold=image_threshold, k=shortlist_k
        )
        logging.info(f"Image prepass matching completed in {time.perf_counter() - prepass_t0:.2f}s")

        #logging.debug(f"Image prepass matched: {len(image_rows)}")

    # --- GUARDED MATCH ---
    guarded_t0 = time.perf_counter()
    logging.info("Starting guarded table/text/image matcher")
    _emit_progress(progress_sink, task_id, "Comparing text and tables", 38, "text")
    guarded_max_ratio = 38

    async def _run_guarded_with_heartbeat() -> Dict[str, Any]:
        nonlocal guarded_max_ratio
        heartbeat_interval_seconds = 20.0
        heartbeat_stop = asyncio.Event()
        text_scoring_started = False
        text_scoring_first_checkpoint_seen = False

        def _guarded_progress_sink(event: Dict[str, Any]) -> None:
            nonlocal guarded_max_ratio, text_scoring_started, text_scoring_first_checkpoint_seen
            if progress_sink is None:
                return
            payload = dict(event)
            message = str(payload.get("message") or "")
            lowered = message.lower()
            if lowered in {"textvectorizer scoring started for text", "reviewing text sections"}:
                text_scoring_started = True
            elif lowered.startswith("textvectorizer scoring text:") or lowered.startswith("reviewing text sections:"):
                text_scoring_first_checkpoint_seen = True
            try:
                incoming_ratio = int(payload.get("completed_ratio", guarded_max_ratio))
            except Exception:
                incoming_ratio = guarded_max_ratio
            clamped_ratio = max(guarded_max_ratio, min(80, incoming_ratio))
            guarded_max_ratio = clamped_ratio
            payload["completed_ratio"] = clamped_ratio
            if not payload.get("task_id") and task_id:
                payload["task_id"] = task_id
            progress_sink(payload)

        async def _guarded_heartbeat() -> None:
            nonlocal guarded_max_ratio, text_scoring_started, text_scoring_first_checkpoint_seen
            while not heartbeat_stop.is_set():
                await asyncio.sleep(heartbeat_interval_seconds)
                if heartbeat_stop.is_set():
                    break
                if not text_scoring_started or text_scoring_first_checkpoint_seen:
                    continue
                elapsed = int(time.perf_counter() - guarded_t0)
                # Advance slowly during long text scoring gaps without regressing emitted ratios.
                heartbeat_ratio = max(guarded_max_ratio, min(79, 38 + (elapsed // 20)))
                guarded_max_ratio = heartbeat_ratio
                _emit_progress(
                    _guarded_progress_sink,
                    task_id,
                    f"Still comparing text and tables ({elapsed}s elapsed)",
                    heartbeat_ratio,
                    "text",
                )

        hb_task = asyncio.create_task(_guarded_heartbeat())
        try:
            logging.info(f"Starting guarded matcher worker thread for task_id={task_id}")
            return await asyncio.to_thread(
                match_one_revision_guarded,
                A,
                B,
                thresholds,
                policy,
                task_id,
                38,
                80,
                _guarded_progress_sink,
            )
        finally:
            heartbeat_stop.set()
            hb_task.cancel()
            await asyncio.gather(hb_task, return_exceptions=True)

    res_guard = asyncio.run(_run_guarded_with_heartbeat())
    logging.info(f"Guarded matcher completed in {time.perf_counter() - guarded_t0:.2f}s")
    _emit_progress(progress_sink, task_id, "Text and table comparison completed", 80, "text")

    # --- Inject image results ---
    if "image" in kinds:
        res_guard["matches"]["image"] = image_rows
        res_guard["unmatched"]["image"] = image_unmatched
        res_guard["summary"]["matched"]["image"] = len(image_rows)
        res_guard["summary"]["source_counts"]["image"] = sum(1 for a in A if a.type == "image")
        res_guard["summary"]["revision_counts"]["image"] = sum(1 for b in B if b.type == "image")
    else:
        res_guard["matches"].setdefault("image", [])
        res_guard["unmatched"].setdefault("image", {"source": [], "revision": []})
        res_guard["summary"]["matched"]["image"] = 0
        res_guard["summary"]["source_counts"]["image"] = 0
        res_guard["summary"]["revision_counts"]["image"] = 0

    # --- Artifact row builder (optimized lookup) ---
    def _row(a):
        chosen_text = getattr(a, "v1_text", None) or getattr(a, "v2_text", None) or getattr(a, "text_orig", "") or ""

        return {
            "id": a.id,
            "type": a.type,
            "page_idx": a.page_idx,
            "parent": a.parent,
            "text": chosen_text,
            "img_path": a.img_path,
            "table_body": a.html if a.type == "table" else None,
            "image_text_extracted": getattr(a, "_img_text_final", "") if a.type == "image" else "",
            "order": getattr(a, "order", 0),
        }

    # --- Build artifacts lookup (fast dict comprehension) ---
    artifacts = {
        "source": {a.id: _row(a) for a in A},
        "revision": {b.id: _row(b) for b in B},
    }

    res_guard["artifacts"] = artifacts

    logging.info(f"match_revision_image_first completed in {time.perf_counter() - stage_t0:.2f}s")

    return res_guard
