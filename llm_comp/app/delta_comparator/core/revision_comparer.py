# revision_comparer.py  — drop-in, self-contained
from __future__ import annotations
import pandas as pd
import os, re, json, asyncio, difflib
from typing import Any, Dict, List, Tuple, Optional
from bs4 import BeautifulSoup
# external deps used by your project
try:
    from openai import AsyncAzureOpenAI
except Exception:
    # adjust import path if your client is elsewhere
    from openai import AsyncAzureOpenAI  # keep same for your env
from rabbitMQ_manager import RabbitMQManager, create_queue_publish


try:
    from loguru import logger as loguru
except Exception:
    class _Null:
        def __getattr__(self, *_): 
            def f(*a, **k): pass
            return f
    loguru = _Null()

from app.delta_comparator.core.prompt import MULTI_PAIR_PROMPT

############################################################
# Utility normalization focused on CONTENT
############################################################

# --- Table parsing & canonicalization ---
_TR_TAG = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.I | re.S)
_TD_TAG = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.I | re.S)

############# Hallucination Pruning Functions
def _extract_row_keys_from_table(text: str, html: str) -> set[str]:
    """
    Extract row_keys from Version A canonical table.
    Row key = first non-empty cell in each data row.
    """
    keys = set()
    matrix = _table_matrix(text, html)
    if not matrix:
        return keys

    header, rows = _header_and_rows(matrix)
    for row in rows:
        rk = _first_non_empty(row)
        if rk:
            keys.add(str(rk).strip())

    return keys

def _prune_invalid_deleted_rows(
    item: Dict[str, Any],
    valid_version_a_keys: set[str]
):
    """
    Remove deleted_rows whose row_key never existed in Version A.
    """
    if not valid_version_a_keys:
        # If Version A had no canonical rows, nothing can be deleted
        item["deleted_rows"] = []
        return

    cleaned = []
    for row in item.get("deleted_rows", []):
        rk = str(row.get("row_key", "")).strip()
        if rk and rk in valid_version_a_keys:
            cleaned.append(row)

    item["deleted_rows"] = cleaned

def _strip_tags_keep_text(s: str) -> str:
    if not s: return ""
    t = re.sub(r"(?i)<br\s*/?>", "\n", s)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _parse_html_table(html: str) -> List[List[str]]:
    if not html: return []
    rows = []
    for tr in _TR_TAG.findall(html or ""):
        cells = _TD_TAG.findall(tr)
        if not cells: 
            continue
        row = [_strip_tags_keep_text(c) for c in cells]
        rows.append(row)
    return rows

def _parse_md_table(md: str) -> List[List[str]]:
    if not md: return []
    lines = [ln.strip().strip("`") for ln in str(md).splitlines() if "|" in ln]
    if not lines: return []
    out = []
    for ln in lines:
        s = ln
        if s.startswith("|"): s = s[1:]
        if s.endswith("|"): s = s[:-1]
        cells = [re.sub(r"\s+", " ", c.strip()) for c in s.split("|")]
        # skip separator rows like ---|:---|---
        if len(cells) >= 1 and all(re.match(r"^[\s:\-]+$", c) for c in cells if c):
            continue
        out.append(cells)
    return out

def _table_matrix(text: str, html: str) -> List[List[str]]:
    # priority: explicit HTML table, else markdown pipe table detected in text
    if html and "<table" in html.lower():
        m = _parse_html_table(html)
        if m: return m
    if text and "|" in text:
        m = _parse_md_table(text)
        if m: return m
    return []

def _header_and_rows(matrix: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    if not matrix: return [], []
    header = matrix[0]
    rows = matrix[1:] if len(matrix) > 1 else []
    return header, rows

def _first_non_empty(vals: List[str]) -> str:
    for v in vals:
        if v and str(v).strip():
            return str(v).strip()
    return ""

def _canonical_table_repr(header: List[str], rows: List[List[str]]) -> str:
    """A simple flat string used for fast equality; ignores spacing differences."""
    h = " | ".join([c.strip().lower() for c in header])
    r = [" | ".join([c.strip().lower() for c in row]) for row in rows]
    return h + "\n" + "\n".join(r)

_MD_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)

def _strip_md_formatting(s: str) -> str:
    if not s:
        return ""
    t = str(s)
    # keep code content; drop fences
    t = _MD_FENCE.sub(r"\1", t)
    # bold/italic/underline/strike/inline-code markers
    t = re.sub(r"`+([^`]+?)`+", r"\1", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"\*(.+?)\*", r"\1", t)
    t = re.sub(r"__(.+?)__", r"\1", t)
    t = re.sub(r"_(.+?)_", r"\1", t)
    t = re.sub(r"~~(.+?)~~", r"\1", t)
    # headers
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)
    return t

def _table_like(s: str) -> bool:
    if not s or not str(s).strip():
        return False
    t = str(s).strip()
    if re.search(r"<table\b", t, flags=re.IGNORECASE):
        return True
    if "|" in t:
        lines = [ln for ln in t.splitlines() if ln.strip()]
        if len(lines) >= 2 and re.match(r"^\s*[:\-\s\|]+\s*$", lines[1]):
            return True
        if lines and lines[0].count("|") >= 1 and len(lines[0].split("|")) >= 2:
            return True
    return False

def _normalize_for_content_comparison(s: str) -> str:
    if not s:
        return ""
    t = _strip_md_formatting(s)
    # html -> remove tags but keep inner text; keep <br> as newline
    #t = re.sub(r"(?i)<br\s*/?>", "\n", t)
    t = re.sub(r"(?i)<br\s*/?>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)    
    # table pipes -> space
    t = re.sub(r"\s*\|\s*", " ", t)
    # compact 2 5 -> 25, normalize <= >= and units
    t = re.sub(r"(?<=\d)\s+(?=\d)", "", t)
    t = re.sub(r"<\s*=", "<=", t); t = re.sub(r">\s*=", ">=", t)
    t = re.sub(r"\b(km)\s*/\s*(h)\b", r"\1/h", t, flags=re.I)
    t = re.sub(r"\b(m)\s*/\s*(s)\b", r"\1/s", t, flags=re.I)
    # punctuation noise -> space (keep / already normalized)
    t = re.sub(r"[^0-9a-zA-Z\s/%+\-<>=]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def _jaccard_identical(a: str, b: str) -> bool:
    aw = set((a or "").split())
    bw = set((b or "").split())
    if not aw and not bw:
        return False
    return len(aw & bw) / max(1, len(aw | bw)) == 1.0

def _seq_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

############################################################
# Comparer
############################################################
class RevisionComparer:
    def __init__(self):
        # Azure client
        base_url = os.getenv("AZURE_ENDPOINT")
        api_key = os.getenv("AZURE_API_KEY")
        self.model_name = os.getenv("AZURE_MODEL_GPT")
        self.temperature = float(os.getenv("VLLM_MODEL_TEMPERATURE", "0.0") or 0.0)
        self.max_tokens = int(os.getenv("TRACER_MAX_TOKENS", "1500"))
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version="2024-05-01-preview",
            azure_endpoint=base_url
        )

        self.MEDIA_EXTENSIONS = (".png",".jpg",".jpeg",".gif",".bmp",".wmf",".svg",".mp4",".avi",".mov",".webm")
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self._response_publish_tasks: set[asyncio.Task] = set()

    async def _publish_response_event(self, task_id: str, result: Dict[str, Any]) -> None:
        queue_name = f"delta_comparator_{task_id}"
        try:
            await create_queue_publish(queue_name, result)
        except Exception as publish_error:
            loguru.warning(f"Dropped response stream event for task {task_id}: {publish_error}")

    async def _publish_stream_event(self, task_id: str, result: Dict[str, Any]) -> None:
        queue_name = f"delta_comparator_{task_id}"
        try:
            await create_queue_publish(queue_name, result)
        except Exception as publish_error:
            loguru.warning(f"Dropped tracer stream event for task {task_id}: {publish_error}")

    def _schedule_response_publish(self, task_id: str, result: Dict[str, Any]) -> None:
        task = asyncio.create_task(self._publish_response_event(task_id, result))
        self._response_publish_tasks.add(task)
        task.add_done_callback(self._response_publish_tasks.discard)

    def _schedule_stream_publish(self, task_id: str, result: Dict[str, Any]) -> None:
        task = asyncio.create_task(self._publish_stream_event(task_id, result))
        self._response_publish_tasks.add(task)
        task.add_done_callback(self._response_publish_tasks.discard)

    async def _flush_response_publish_tasks(self) -> None:
        if not self._response_publish_tasks:
            return
        pending_tasks = list(self._response_publish_tasks)
        self._response_publish_tasks.clear()
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    #############################
    # Fast identical check
    #############################
    def if_modified(self, left: str, right: str, kind: str, left_html: str = "", right_html: str = "") -> bool:
        """
        Return True if effectively identical (No Change). False otherwise.
        Tables: compare canonical matrices (header+rows) to ignore formatting noise.
        Text/Image: use normalized content comparison.
        """
        if kind == "table":
            L = _table_matrix(left, left_html)
            R = _table_matrix(right, right_html)
            if not L and not R:
                # fall back to text normalization
                l = _normalize_for_content_comparison(left or "")
                r = _normalize_for_content_comparison(right or "")
                return (not l and not r) or l == r or _jaccard_identical(l, r)
            Lh, Lr = _header_and_rows(L)
            Rh, Rr = _header_and_rows(R)
            return _canonical_table_repr(Lh, Lr) == _canonical_table_repr(Rh, Rr)
        else:
            l = _normalize_for_content_comparison(left or "")
            r = _normalize_for_content_comparison(right or "")
            if not l and not r:
                return True
            if l == r:
                return True
            return _jaccard_identical(l, r)


    def is_media_content(self, s: str) -> bool:
        if not isinstance(s, str): return False
        tl = s.lower()
        if any(ext in tl for ext in self.MEDIA_EXTENSIONS):
            return True
        return bool(re.search(r"\.(png|jpg|jpeg|gif|bmp|wmf|svg|mp4|avi|mov|webm)(?:[\)\]\s]|$)", tl, re.I))

    def classify_trivial_pair(self, left: str, right: str, kind: str) -> Tuple[str, str] | None:
        """
        Returns ("No Change"|"New"|"Deleted"|"Skipped", description) or None (non-trivial).
        """
        l = str(left or "").strip()
        r = str(right or "").strip()

        # both empty: no change
        if not l and not r:
            return ("No Change", "Both sides empty")
        # both trivial media refs and no text:
        if self.is_media_content(l) and self.is_media_content(r) and not (left or "").strip() and not (right or "").strip():
            return ("Skipped", "Media-only placeholders")
        # additions/deletions
        if not l and r:
            return ("New", "Added content")
        if l and not r:
            return ("Deleted", "Removed content")

        # formatting-only quick test (sequence ratio on normalized)
        nl = _normalize_for_content_comparison(l)
        nr = _normalize_for_content_comparison(r)
        if nl == nr or _seq_ratio(nl, nr) >= 0.98:
            return ("No Change", "Formatting-only differences")

        # for very short text lines, avoid LLM if near-identical
        if kind == "text" and min(len(nl.split()), len(nr.split())) <= 6:
            if _seq_ratio(nl, nr) >= 0.96:
                return ("No Change", "Short-line near-identical")

        return None

    #############################
    # LLM plumbing
    #############################
    def _format_rules(self, lst: List[str]) -> str:
        if not lst: return ""
        return "\n- " + "\n- ".join(lst)

    def _multi_pair_prompt(self, batch: List[Tuple[str, str, str, str, str]], rules: dict) -> str:
        """
        batch: list of (pair_key, kind, row_id, left_blob, right_blob)
        left_blob/right_blob are raw-ish content (markdown/HTML for tables; IMAGE_TEXT:... for images)
        """
        parts = []
        for pair_key, kind, row_id, left_blob, right_blob in batch:
            parts.append(
                f'PAIR_KEY: "{pair_key}"\nKIND: "{kind}"\nROW_ID: "{row_id}"\n---\nVersion A:\n{left_blob}\n---\nVersion B:\n{right_blob}\n---'
            )
        pairs_block = "\n\n".join(parts)
        return MULTI_PAIR_PROMPT.format(
            MAJOR_RULES=self._format_rules(rules.get("major_rules", [])),
            MINOR_RULES=self._format_rules(rules.get("minor_rules", [])),
            NO_CHANGE_RULES=self._format_rules(rules.get("no_change_rules", [])),
            PAIRS_BLOCK=pairs_block
        )

    async def _call_llm_batch(self, prompt: str, pair_keys: List[str]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
        resp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert requirement change classifier."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,   # For GPT 4.0
            #max_completion_tokens=self.max_tokens FOR GPT 5
        )
        raw = (getattr(resp.choices[0].message, "content", "") or "").strip()

        # track usage if present
        usage = getattr(resp, "usage", None)
        used = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if usage:
            used = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }

        # strip fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.I).strip("` \n\r\t")

        def try_json(s: str):
            try:
                return json.loads(s)
            except Exception:
                return None

        parsed = None
        if cleaned.startswith("[") or cleaned.startswith("{"):
            parsed = try_json(cleaned)
        if parsed is None:
            m = re.search(r"(\[.*\])", cleaned, flags=re.DOTALL)
            if m: parsed = try_json(m.group(1))
        if parsed is None:
            objs = re.findall(r"(\{(?:[^{}]|\{[^{}]*\})*?\})", cleaned, flags=re.DOTALL)
            if objs:
                parsed = try_json("[" + ",".join(objs) + "]")

        out: Dict[str, Dict[str, Any]] = {}
        if not isinstance(parsed, list):
            # map all to Unidentified
            for pk in pair_keys:
                out[pk] = {"change_type":"Unidentified","change_description":"Failed to parse LLM output"}
            return out, used

        # normalize per object
        for item in parsed:
            if not isinstance(item, dict): continue
            pk = str(item.get("pair_key",""))
            if not pk: continue
            # normalize fields
            if "description" in item and "change_description" not in item:
                item["change_description"] = item["description"]
            if "change_description" in item and "description" not in item:
                item["description"] = item["change_description"]
            if "row_changes" not in item: item["row_changes"] = []
            out[pk] = item

        return out, used
    
    def get_stream_response_ready_llm(self, task_id, df, version_nums, n_versions, entry_data, completed_ratio_llm: int | None = None):
        result = None
        for row_idx, row in df.iterrows():
            kind = (row.get("type") or "").strip().lower() or "text"
            if kind not in ("text", "table", "image"):
                kind = "text"

            version_data = {}
            for v in version_nums:
                version_data[f"v{v}"] = {
                    "text": row.get(f"v{v}_text", ""),
                    "img_path": row.get(f"v{v}_img_path", ""),
                    "table_body": row.get(f"v{v}_table_body", "")
                }

            change_log = {}
            row_level_changes = {}
            for i in range(n_versions - 1):
                vA, vB = version_nums[i], version_nums[i + 1]
                pk = f"{row_idx}||v{vA}-v{vB}||WHOLE"
                entry = entry_data.get(pk) 
                if entry:
                    pair_name = f"v{vA}-v{vB}"
                    change_log[pair_name] = {"pairs": [entry]}
                    row_level_changes[pair_name] = [entry]
            if row_level_changes:
                result = {
                    "type": "response",
                    "task_id": task_id,
                    "kind": kind,
                    "completed_ratio_llm": completed_ratio_llm,
                    "versions": version_data,
                    "Change_log": change_log,
                    "Row_level_changes": row_level_changes
                }
                self._schedule_response_publish(task_id, result)
                # print(f"\n\n\n\n change delta for llm ---------{result}")
            return result



    def get_stream_response_ready(self, task_id, row, vA, vB, pair_key, version_nums, n_versions, entry_data, completed_ratio_llm: int | None = None):
        result = {}
        kind = (row.get("type") or "").strip().lower() or "text"
        if kind not in ("text", "table", "image"):
            kind = "text"

        version_data = {}
        for v in version_nums:
            version_data[f"v{v}"] = {
                "text": row.get(f"v{v}_text", ""),
                "img_path": row.get(f"v{v}_img_path", ""),
                "table_body": row.get(f"v{v}_table_body", "")
            }

        change_log = {}
        row_level_changes = {}
        for i in range(n_versions - 1):
            entry = entry_data.get(pair_key)  
            if entry:
                pair_name = f"v{vA}-v{vB}"
                change_log[pair_name] = {"pairs": [entry]}
                row_level_changes[pair_name] = [entry]

        result =  {
                "type": "response",
                "task_id": task_id,
                "kind": kind,
                "completed_ratio_llm": completed_ratio_llm,
                "versions": version_data,
                "Change_log": change_log,
                "Row_level_changes": row_level_changes
            }
        self._schedule_response_publish(task_id, result)
        return result


    #############################
    # Core runner
    #############################

    async def run_and_log(self,
                      df: pd.DataFrame,
                      rules: dict,
                      task_id: str,
                      artifact_path: str = "artifacts/change_trace.json",
                      batch_size: int = 6,
                      max_concurrency: int = 6,
                      **kwargs) -> Tuple[Dict[str, Any], int]:
        """
        N-version comparison:
        - Detects available versions (v1...vN) from columns.
        - Compares consecutive pairs: v1-v2, v2-v3, ...
        - Retains vX_text, vX_img_path, vX_table_body in output.
        Adds detailed logging for queueing, trivial vs LLM, batching, timings, and token usage.
        """
        from app.delta_comparator.utils.logger import log as logging
        import time

        t_start = time.perf_counter()

        # --- Detect versions ---
        #logging.debug(f"Detecting versions from DataFrame columns...{len(df.columns)} columns found. {df.columns.tolist()}")
        version_cols = [c for c in df.columns if re.match(r"v\d+_text$", c)]
        version_nums = sorted({int(re.findall(r"\d+", c)[0]) for c in version_cols})
        n_versions = len(version_nums)
        if n_versions < 2:
            raise ValueError("Need at least 2 versions (v1,v2,...) to compare")
        # for v in version_nums:
        #     #logging.debug(f"Detected versions: v{v}")
        # #logging.debug(f"Rows to process: {len(df)}")

        # --- Counters for telemetry ---
        total_pairs_possible = (n_versions - 1) * len(df)
        cnt_trivial_nochange = 0
        cnt_trivial_other = 0
        cnt_llm_sent = 0
        cnt_tables_identical = 0

        # --- Prepare pending and trivial results ---
        pending: List[Tuple[str, str, str, str, str]] = []
        trivial_by_pk: Dict[str, Dict[str, Any]] = {}
        pair_keys_by_row: Dict[int, List[str]] = {}
        queue_name = f"delta_comparator_{task_id}"

        t_queue = time.perf_counter()
        completed = 0
        latest_progress_kind = "text"
        last_response_publish_at = 0.0
        response_publish_interval_seconds = 8
        latest_response_payload: Dict[str, Any] | None = None
        tracer_stream_start_ratio = 80
        tracer_stream_end_ratio = 90
        tracer_stream_span = tracer_stream_end_ratio - tracer_stream_start_ratio
        last_stream_progress_ratio = tracer_stream_start_ratio
        emit_tracer_running_stream = False

        def mark_progress(kind_name: str):
            nonlocal latest_progress_kind
            latest_progress_kind = kind_name

        def should_publish_response() -> bool:
            nonlocal last_response_publish_at
            now = time.perf_counter()
            if (now - last_response_publish_at) < response_publish_interval_seconds:
                return False
            last_response_publish_at = now
            return True

        def get_completed_ratio_llm() -> int:
            return int(completed / total_pairs_possible * 100) if total_pairs_possible else 100

        def get_stream_completed_ratio() -> int:
            llm_ratio = min(get_completed_ratio_llm(), 100)
            return tracer_stream_start_ratio + int((tracer_stream_span * llm_ratio) / 100)

        def publish_tracer_stream_update() -> None:
            if not emit_tracer_running_stream:
                return
            nonlocal last_stream_progress_ratio
            completed_ratio = get_stream_completed_ratio()
            if completed_ratio <= last_stream_progress_ratio:
                return
            last_stream_progress_ratio = completed_ratio
            self._schedule_stream_publish(
                task_id,
                {
                    "type": "stream",
                    "task_id": task_id,
                    "kind": latest_progress_kind,
                    "completed_ratio": completed_ratio,
                    "message": f"Change tracer running: {get_completed_ratio_llm()}% complete",
                },
            )

        def register_trivial_result(row_idx, row, vA, vB, pair_key, entry, kind_name: str):
            nonlocal completed, latest_response_payload
            trivial_by_pk[pair_key] = entry
            pair_keys_by_row.setdefault(row_idx, []).append(pair_key)
            completed += 1
            mark_progress(kind_name)
            if should_publish_response():
                latest_response_payload = self.get_stream_response_ready(task_id, row, vA, vB, pair_key, version_nums, n_versions, trivial_by_pk, completed_ratio_llm=get_completed_ratio_llm())
                publish_tracer_stream_update()
         ####Functionss for cleaning table data before sending to LLM to compare
        def html_table_to_matrix(html):            
            soup = BeautifulSoup(html or "", "html.parser")                               
            matrix = []
            trs = soup.find_all("tr")
            if not trs:
                return []
            for tr in trs:
                cells = tr.find_all(["td", "th"])
                row = [c.get_text(strip=True) if c else "" for c in cells]
                # skip fully empty rows
                if any(cell.strip() for cell in row):
                    matrix.append(row)
            return matrix
        
        
        def matrix_to_markdown(matrix):
            # if empty matrix, return empty markdown
            if not matrix or len(matrix) == 0:
                return ""

            # normalize: ensure all rows have equal column count
            max_cols = max(len(row) for row in matrix)
            canonical = []
            for row in matrix:
                padded = row + [""] * (max_cols - len(row))
                canonical.append(padded)
            header = canonical[0]
            body = canonical[1:]
            header_line = "| " + " | ".join(header) + " |"
            separator_line = "| " + " | ".join(["---"] * len(header)) + " |"
            body_lines = ["| " + " | ".join(r) + " |" for r in body]
            return "\n".join([header_line, separator_line] + body_lines)
        t_queue = time.perf_counter()
        for row_idx, row in df.iterrows():
            kind = (row.get("type") or "").strip().lower() or "text"
            if kind not in ("text", "table", "image"):
                kind = "text"

            for i in range(n_versions - 1):
                vA, vB = version_nums[i], version_nums[i + 1]
                left_text   = str(row.get(f"v{vA}_text", "") or "")
                right_text  = str(row.get(f"v{vB}_text", "") or "")
                left_html   = str(row.get(f"v{vA}_table_body", "") or "")
                right_html  = str(row.get(f"v{vB}_table_body", "") or "")
                left_img    = str(row.get(f"v{vA}_img_path", "") or "")
                right_img   = str(row.get(f"v{vB}_img_path", "") or "")

                pair_key = f"{row_idx}||v{vA}-v{vB}||WHOLE"

                # Build blobs + compare
                if kind == "table":
                    left_for_compare = left_text
                    right_for_compare = right_text

                    # canonical identical?
                    if self.if_modified(left_for_compare, right_for_compare, kind, left_html, right_html):
                        cnt_tables_identical += 1
                        cnt_trivial_nochange += 1
                        register_trivial_result(
                            row_idx,
                            row,
                            vA,
                            vB,
                            pair_key,
                            {
                            "pair_key": pair_key,
                            "change_type": "No Change",
                            "change_description": "Identical table content (canonicalized)",
                            "formatting_only": True,
                            "row_changes": [],
                            "added_rows": [],
                            "deleted_rows": []
                            },
                            kind,
                        )
                        continue

                    # quick trivial (empty/new/deleted/formatting-only)
                    tri = self.classify_trivial_pair(left_for_compare, right_for_compare, kind)
                    if tri:
                        (ctype, cdesc) = tri
                        cnt_trivial_nochange += (1 if ctype == "No Change" else 0)
                        cnt_trivial_other += (0 if ctype == "No Change" else 1)
                        register_trivial_result(
                            row_idx,
                            row,
                            vA,
                            vB,
                            pair_key,
                            {
                            "pair_key": pair_key,
                            "change_type": ctype,
                            "change_description": cdesc,
                            "formatting_only": (ctype == "No Change"),
                            "row_changes": [],
                            "added_rows": [],
                            "deleted_rows": []
                            },
                            kind,
                        )
                        continue

                    # left_blob  = f"TABLE_RAW:\n{left_for_compare}\n"
                    # right_blob = f"TABLE_RAW:\n{right_for_compare}\n"
                    left_blob  = f"TABLE_RAW:\n{left_html}\n"
                    right_blob = f"TABLE_RAW:\n{right_html}\n"

                elif kind == "image":
                    # prefer extracted text; include path cue
                    l_txt = left_text
                    r_txt = right_text
                    left_for_compare, right_for_compare = l_txt, r_txt
                    # left_blob  = f"IMAGE_TEXT: {l_txt}\nIMAGE_PATH: {left_img}"
                    # right_blob = f"IMAGE_TEXT: {r_txt}\nIMAGE_PATH: {right_img}"
                    left_blob  = f"IMAGE_TEXT: {l_txt}"
                    right_blob = f"IMAGE_TEXT: {r_txt}"

                else:  # text
                    left_for_compare, right_for_compare = left_text, right_text
                    left_blob, right_blob = left_for_compare, right_for_compare

                # identical?
                if self.if_modified(left_for_compare, right_for_compare, kind, left_html, right_html):
                    cnt_trivial_nochange += 1
                    register_trivial_result(
                        row_idx,
                        row,
                        vA,
                        vB,
                        pair_key,
                        {
                        "pair_key": pair_key,
                        "change_type": "No Change",
                        "change_description": "Identical content",
                        "formatting_only": True,
                        "row_changes": []
                        },
                        kind,
                    )
                    continue

                # trivial classifier
                tri = self.classify_trivial_pair(left_for_compare, right_for_compare, kind)
                if tri:
                    (ctype, cdesc) = tri
                    cnt_trivial_nochange += (1 if ctype == "No Change" else 0)
                    cnt_trivial_other += (0 if ctype == "No Change" else 1)
                    register_trivial_result(
                        row_idx,
                        row,
                        vA,
                        vB,
                        pair_key,
                        {
                        "pair_key": pair_key,
                        "change_type": ctype,
                        "change_description": cdesc,
                        "formatting_only": (ctype == "No Change"),
                        "row_changes": []
                        },
                        kind,
                    )
                    continue

                # enqueue for LLM
                pending.append((pair_key, kind, str(row_idx), left_blob, right_blob))
                pair_keys_by_row.setdefault(row_idx, []).append(pair_key)

        t_queue_end = time.perf_counter()
        #logging.debug(f"Queueing finished in {(t_queue_end - t_queue):.3f}s")
        #logging.debug(f"Total possible pairs: {total_pairs_possible} | trivial_nochange={cnt_trivial_nochange} | trivial_other={cnt_trivial_other} | queued_for_LLM={len(pending)}")

        # --- Batch LLM calls ---
        results_from_llm: Dict[str, Dict[str, Any]] = {}
        if pending:
            semaphore = asyncio.Semaphore(max_concurrency)

            async def process_slice(slice_pairs: List[Tuple[str, str, str, str, str]], batch_id: int):
                nonlocal completed, latest_response_payload
                async with semaphore:
                    t0 = time.perf_counter()
                    pair_keys = [p[0] for p in slice_pairs]
                    prompt = self._multi_pair_prompt(slice_pairs, rules or {})
                    out_map, usage = await self._call_llm_batch(prompt, pair_keys)
                    results_from_llm.update(out_map)
                    completed += len(slice_pairs)
                    mark_progress("llm")
                    if should_publish_response():
                        latest_response_payload = self.get_stream_response_ready_llm(task_id, df, version_nums, n_versions, out_map, completed_ratio_llm=get_completed_ratio_llm()) or latest_response_payload
                        publish_tracer_stream_update()

                    self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                    self.total_completion_tokens += usage.get("completion_tokens", 0)
                    self.total_tokens += usage.get("total_tokens", 0)
                    pt = usage.get("prompt_tokens", 0)
                    ct = usage.get("completion_tokens", 0)
                    tt = usage.get("total_tokens", 0)
                    log_time = time.perf_counter() - t0
                    #logging.debug(f"LLM batch #{batch_id}: pairs={len(slice_pairs)}, prompt_tokens={pt}, completion_tokens={ct}, total_tokens={tt}, time={log_time}s")

            tasks = []
            i = 0
            batch_id = 1

            TABLE_BATCH_SIZE = 3   # Control Tble batch size

            while i < len(pending):
                current = pending[i]

                if current[1] == "table":
                    #  collect up to TABLE_BATCH_SIZE tables
                    #logging.debug(f"Batching for tables: starting at index {i} (pair_key={current[0]})")
                    table_batch = []
                    j = i
                    while j < len(pending) and len(table_batch) < TABLE_BATCH_SIZE:
                        if pending[j][1] == "table":
                            table_batch.append(pending[j])
                        else:
                            break  # stop if non-table encountered
                        j += 1

                    batch_pairs = table_batch
                    i += len(table_batch)

                else:
                    # normal batching for text/image
                    #logging.debug(f"Batching for text/image: starting at index {i} (pair_key={current[0]})")
                    batch_pairs = []
                    j = i
                    while j < len(pending) and len(batch_pairs) < batch_size:
                        if pending[j][1] != "table":
                            batch_pairs.append(pending[j])
                            j += 1
                        else:
                            break  # stop when table encountered

                    if not batch_pairs:
                        # safety fallback
                        batch_pairs = [current]
                        i += 1
                    else:
                        i += len(batch_pairs)

                tasks.append(asyncio.create_task(process_slice(batch_pairs, batch_id)))
                batch_id += 1
            await asyncio.gather(*tasks)

            cnt_llm_sent = len(pending)
        else:
            logging.debug("No pairs required LLM scoring.")

        # --- Build final JSON in same row order ---
        

        requirements: List[Dict[str, Any]] = []
        for row_idx, row in df.iterrows():
            kind = (row.get("type") or "").strip().lower() or "text"
            if kind not in ("text", "table", "image"):
                kind = "text"

            version_data = {}
            for v in version_nums:
                version_data[f"v{v}"] = {
                    "text": row.get(f"v{v}_text", ""),
                    "img_path": row.get(f"v{v}_img_path", ""),
                    "table_body": row.get(f"v{v}_table_body", ""),
                    "page": row.get(f"v{v}_page", None),     
                    "order": row.get(f"v{v}_order", None),   
                }

            change_log = {}
            row_level_changes = {}
            for i in range(n_versions - 1):
                vA, vB = version_nums[i], version_nums[i + 1]
                pk = f"{row_idx}||v{vA}-v{vB}||WHOLE"
                entry = results_from_llm.get(pk) or trivial_by_pk.get(pk) or {
                    "pair_key": pk,
                    "change_type": "Unidentified",
                    "change_description": "No result",
                    "row_changes": [],
                    "added_rows": [],
                    "deleted_rows": []
                }
                if entry.get("change_type") == "Deleted":
                    entry["page_info"] = {
                        f"v{vA}": row.get(f"v{vA}_page", None),
                        f"v{vB}": row.get(f"v{vA}_page", None),
                    }
                else:
                    entry["page_info"] = {
                        f"v{vA}": row.get(f"v{vA}_page", None),
                        f"v{vB}": row.get(f"v{vB}_page", None),
                    }

                # ----------------------------------------
                # HARD STOP: remove hallucinated deletions
                # ----------------------------------------
                if kind == "table":
                    vA = version_nums[i]
                    valid_keys = _extract_row_keys_from_table(
                        str(row.get(f"v{vA}_text", "") or ""),
                        str(row.get(f"v{vA}_table_body", "") or "")
                    )
                    _prune_invalid_deleted_rows(entry, valid_keys)
                # ----------------------------------------

                pair_name = f"v{vA}-v{vB}"
                change_log[pair_name] = {"pairs": [entry]}
                #row_level_changes[pair_name] = [entry]

            requirements.append({
                "kind": kind,
                "versions": version_data,
                "Change_log": change_log                
            })

        final_doc = {
            "requirements": requirements,
            "token_usage": {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens
            },
            "stats": {
                "versions": [f"v{v}" for v in version_nums],
                "rows": len(df),
                "pairs_possible": total_pairs_possible,
                "pairs_trivial_nochange": cnt_trivial_nochange,
                "pairs_trivial_other": cnt_trivial_other,
                "pairs_sent_to_llm": cnt_llm_sent,
                "tables_identical_canonical": cnt_tables_identical
            }
        }

        # --- Persist
        os.makedirs(os.path.dirname(artifact_path) or ".", exist_ok=True)
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(final_doc, f, indent=2, ensure_ascii=False)

        t_end = time.perf_counter()
        # logging.debug(
        #     f"Finished N-version comparison in {(t_end - t_start):.2f}s | pairs_possible={total_pairs_possible} | trivial_nochange={cnt_trivial_nochange} | trivial_other={cnt_trivial_other} | LLM_pairs={cnt_llm_sent} | tokens(total/prompt/comp)={self.total_tokens}/{self.total_prompt_tokens}/{self.total_completion_tokens}")
        # logging.debug(f"Wrote JSON: {artifact_path}")

        # best-effort close
        try:
            if latest_response_payload is not None and latest_response_payload.get("completed_ratio_llm") != 100:
                final_response_payload = dict(latest_response_payload)
                final_response_payload["completed_ratio_llm"] = 100
                self._schedule_response_publish(task_id, final_response_payload)
            await self._flush_response_publish_tasks()
            await self.client.close()
        except Exception:
            pass

        return final_doc, self.total_tokens