# report_html.py
# Generates the html file for the corresponding json file (multi-version aware)
import os
import re
import json
import shutil
from typing import Dict, Any, List, Tuple, Union, Optional
from difflib import SequenceMatcher
from html import escape as html_escape

# -----------------------
# basic helpers
# -----------------------
def _normalize_whitespace(s: Optional[str]) -> str:
    s = "" if s is None else str(s)
    return " ".join(s.split())

def _word_tokens(s: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", s, re.UNICODE)

def _inline_diff_html(a: str, b: str) -> str:
    """Inline diff for TEXT ONLY (not used inside tables)."""
    a_tokens = _word_tokens(a or "")
    b_tokens = _word_tokens(b or "")
    sm = SequenceMatcher(None, a_tokens, b_tokens)
    out: List[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.extend(html_escape(tok) for tok in a_tokens[i1:i2])
        elif tag == "insert":
            seg = " ".join(html_escape(t) for t in b_tokens[j1:j2])
            if seg:
                out.append(f'<span class="ins">{seg}</span>')
        elif tag == "delete":
            seg = " ".join(html_escape(t) for t in a_tokens[i1:i2])
            if seg:
                out.append(f'<span class="del">{seg}</span>')
        elif tag == "replace":
            seg_del = " ".join(html_escape(t) for t in a_tokens[i1:i2])
            seg_ins = " ".join(html_escape(t) for t in b_tokens[j1:j2])
            if seg_del:
                out.append(f'<span class="del">{seg_del}</span>')
            if seg_ins:
                out.append(f'<span class="ins">{seg_ins}</span>')
    return " ".join(out)

def _get_version_keys_from_dict(vdict: Dict[str, Any]) -> List[str]:
    """Return version keys in numeric order: v1, v2, v3... (fallback to alpha for odd keys)."""
    if not isinstance(vdict, dict):
        return []
    vkeys = [k for k in vdict.keys() if isinstance(k, str)]
    if not vkeys:
        return []
    numeric_pairs: List[Tuple[int, str]] = []
    non_numeric: List[str] = []
    for k in vkeys:
        m = re.findall(r"\d+", k)
        if m:
            try:
                numeric_pairs.append((int(m[0]), k))
            except Exception:
                non_numeric.append(k)
        else:
            non_numeric.append(k)
    numeric_pairs.sort(key=lambda t: t[0])
    return [k for _, k in numeric_pairs] + sorted(non_numeric)

def _version_pairs_from_change_log(change_log: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if isinstance(change_log, dict):
        for pid in change_log.keys():
            if isinstance(pid, str) and "-" in pid:
                left, right = pid.split("-", 1)
                left, right = left.strip(), right.strip()
                if left and right:
                    pairs.append((left, right))
    return pairs

# -----------------------
# permissive markdown/table -> HTML converter
# -----------------------
def _markdown_table_to_html(md: str) -> Optional[str]:
    if not md:
        return None
    s = str(md).strip()
    if not s:
        return None
    if re.search(r"<table\b", s, flags=re.IGNORECASE):
        m = re.search(r"(<table\b.*?>.*?</table>)", s, flags=re.IGNORECASE | re.DOTALL)
        return m.group(1) if m else s
    lines = [ln.rstrip() for ln in s.splitlines() if ln.strip()]
    pipe_lines = [ln for ln in lines if "|" in ln]
    if not pipe_lines:
        return None
    def parse_row(ln: str) -> List[str]:
        t = ln.strip()
        if t.startswith("|"):
            t = t[1:]
        if t.endswith("|"):
            t = t[:-1]
        return [c.strip() for c in t.split("|")]
    header_idx = None
    sep_idx = None
    for i in range(len(pipe_lines) - 1):
        candidate = pipe_lines[i + 1].strip()
        cells = [c.strip() for c in re.split(r"\|", candidate)]
        if all(re.fullmatch(r"[:\- ]{1,}", c) for c in cells if c != ""):
            header_idx = i
            sep_idx = i + 1
            break
    if header_idx is None:
        if len(pipe_lines) >= 2:
            header_idx = 0
            data_lines = pipe_lines[1:]
        else:
            header_idx = None
            data_lines = pipe_lines
    else:
        data_lines = pipe_lines[sep_idx + 1 :]

    out = ["<table>"]
    if header_idx is not None:
        headers = parse_row(pipe_lines[header_idx])
        out.append("<thead><tr>")
        for h in headers:
            out.append(f"<th>{html_escape(h)}</th>")
        out.append("</tr></thead>")
    out.append("<tbody>")
    for ln in data_lines:
        if not ln.strip():
            continue
        cells = parse_row(ln)
        if all(re.fullmatch(r"[:\- ]{1,}", c) for c in cells if c != ""):
            continue
        out.append("<tr>")
        for cell in cells:
            out.append(f"<td>{html_escape(cell)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)

# -----------------------
# extract text, image, table for a version value
# -----------------------
def _extract_text_img_table(v: Any) -> Tuple[str, Optional[str], Optional[str]]:
    text = ""
    img = None
    table_html = None

    def _strip_code_fences(s: str) -> str:
        if not s:
            return ""
        t = str(s).strip()
        m = re.match(r'^\s*```(?:\s*([\w+-]+))?\s*\n(.*)\n```(?:\s*)$', t, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(2).strip()
        m2 = re.match(r'^\s*```(?:\s*([\w+-]+))?\s*(.*?)\s*```\s*$', t, flags=re.IGNORECASE | re.DOTALL)
        if m2:
            return m2.group(2).strip()
        m4 = re.match(r'^\s*`(.*)`\s*$', t, flags=re.DOTALL)
        if m4:
            return m4.group(1).strip()
        return re.sub(r'^\s*(?:markdown|md)\s+', '', t, flags=re.IGNORECASE).strip()

    def _maybe_table_from_string(s: str) -> Optional[str]:
        if not s:
            return None
        s = str(s).strip()
        if not s:
            return None
        if re.search(r"<table\b", s, flags=re.IGNORECASE):
            m = re.search(r"(<table\b.*?>.*?</table>)", s, flags=re.IGNORECASE | re.DOTALL)
            return m.group(1) if m else s
        stripped = _strip_code_fences(s)
        maybe = _markdown_table_to_html(stripped)
        if maybe:
            return maybe
        return None

    if isinstance(v, dict):
        text = v.get("text") or v.get("source") or v.get("revised") or ""
        if text is None:
            text = ""
        # table_body / table_html keys
        table_keys = ("table_html", "table_body", "table", "tableBody", "tableHtml",
                      "source_table", "revised_1_table", "revised_table", "table_body_html")
        for k in table_keys:
            if k in v and isinstance(v[k], str) and v[k].strip():
                tb = v[k].strip()
                if re.search(r"<table\b", tb, flags=re.IGNORECASE):
                    m = re.search(r"(<table\b.*?>.*?</table>)", tb, flags=re.IGNORECASE | re.DOTALL)
                    table_html = m.group(1) if m else tb
                else:
                    maybe = _markdown_table_to_html(tb)
                    table_html = maybe if maybe else tb
                break
        if not table_html and isinstance(text, str) and "|" in text:
            maybe = _markdown_table_to_html(text)
            if maybe:
                table_html = maybe
        # image (collect but may be disabled when rendering)
        for k in ("img_path", "image_path", "image", "img", "imagePath", "imgpath", "thumbnail"):
            val = v.get(k)
            if isinstance(val, str) and val.strip():
                img = val.strip().replace("\\", "/")
                break
    else:
        text = "" if v is None else str(v)
        m = re.search(r"(images/[^\s'\"<>]+)", text)
        if m:
            img = m.group(1).replace("\\", "/")
        table_html = _maybe_table_from_string(text)

    if table_html and "<table" not in table_html.lower():
        maybe = _markdown_table_to_html(table_html)
        if maybe:
            table_html = maybe

    return text, img, table_html

# -----------------------
# ensure image available next to HTML (copy if found) and return a relative src
# -----------------------
def _ensure_img_available(img_path: str, out_html_path: str) -> Optional[str]:
    if not img_path:
        return None
    out_dir = os.path.dirname(os.path.abspath(out_html_path)) or "."
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    basename = os.path.basename(img_path)
    candidate_rel = os.path.join(out_dir, img_path) if not os.path.isabs(img_path) else img_path
    if os.path.exists(candidate_rel) and os.path.isfile(candidate_rel):
        dest = os.path.join(images_dir, basename)
        try:
            if os.path.abspath(candidate_rel) != os.path.abspath(dest):
                shutil.copy2(candidate_rel, dest)
        except Exception:
            try:
                shutil.copy(candidate_rel, dest)
            except Exception:
                pass
        return os.path.join("images", basename)
    candidates = [
        img_path,
        os.path.abspath(img_path),
        os.path.join(os.getcwd(), img_path),
        os.path.join(os.getcwd(), "images", basename),
        os.path.join(out_dir, "images", basename),
    ]
    for c in candidates:
        try:
            if c and os.path.exists(c) and os.path.isfile(c):
                dest = os.path.join(images_dir, basename)
                try:
                    if os.path.abspath(c) != os.path.abspath(dest):
                        shutil.copy2(c, dest)
                except Exception:
                    try:
                        shutil.copy(c, dest)
                    except Exception:
                        pass
                return os.path.join("images", basename)
        except Exception:
            continue
    return img_path

# -----------------------
# normalize/aggregate change_type
# -----------------------
_SEVERITY_MAP = {
    "major": "major",
    "major change": "major",
    "minor": "minor",
    "minor change": "minor",
    "new": "new",
    "addition": "new",
    "added": "new",
    "deleted": "deleted",
    "remove": "deleted",
    "removed": "deleted",
    "no change": "nochange",
    "no_change": "nochange",
    "identical": "nochange",
    "skipped": "skipped",
    "unidentified": "unidentified",
    "modified": "modified",
}
def _normalize_change_type(ct: Optional[str]) -> str:
    if not ct:
        return "unidentified"
    s = str(ct).strip().lower()
    for k, v in _SEVERITY_MAP.items():
        if k in s:
            return v
    if "major" in s:
        return "major"
    if "minor" in s:
        return "minor"
    if "new" in s or "add" in s:
        return "new"
    if "del" in s or "remove" in s:
        return "deleted"
    return "unidentified"

def _aggregate_ct_from_rows(per_row_list: List[Dict[str, Any]]) -> str:
    """Aggregate a single change_type from a row list (priority)."""
    if not per_row_list:
        return "unidentified"
    labels = []
    for r in per_row_list:
        if not isinstance(r, dict):
            continue
        labels.append(_normalize_change_type(r.get("change_type")))
    priority = ["major", "minor", "modified", "new", "deleted", "nochange", "skipped", "unidentified"]
    for p in priority:
        if p in labels:
            return p
    return "unidentified"

def _content_differs(a_text: str, a_table: Optional[str], b_text: str, b_table: Optional[str]) -> bool:
    """Plain, conservative content change detection."""
    if a_table or b_table:
        at = _normalize_whitespace(a_table or "")
        bt = _normalize_whitespace(b_table or "")
        return at != bt
    return _normalize_whitespace(a_text) != _normalize_whitespace(b_text)

def _readable_row_label_from_pair_key(pair_key: str) -> str:
    if not pair_key:
        return ""
    parts = [p for p in re.split(r"\|\|", pair_key) if p and p.strip()]
    if not parts:
        return pair_key
    last = parts[-1].strip()
    if re.match(r"(?i)row[_\-]?\d+$", last) and len(parts) >= 2:
        candidate = parts[-2].strip()
    else:
        candidate = last
    candidate = re.sub(r"[_]+", " ", candidate).strip()
    if len(candidate) > 120:
        candidate = candidate[:120] + "…"
    return candidate

def _render_overall_summary_table(summary):
    if not summary:
        return ""

    out = []
    out.append('<div class="adld" id="overall-summary">')
    out.append('<h3>Overall Summary</h3>')

    # Filter buttons
    out.append('<div style="margin-bottom:10px;">')
    for f in ["all", "major", "minor", "modified", "new", "deleted", "nochange"]:
        out.append(
            f'<button onclick="filterSummary(\'{f}\')" '
            f'style="margin-right:6px;padding:4px 10px;cursor:pointer;">'
            f'{f.title()}</button>'
        )
    out.append('</div>')

    out.append('<table id="summary-table">')
    out.append(
        '<thead><tr>'
        '<th>Requirement</th>'
        '<th>Versions</th>'
        '<th>Change Type</th>'
        '<th>WHOLE Description</th>'
        '</tr></thead><tbody>'
    )

    for r in summary:
        display_ct = r["change_type"]
        bucket = r.get("bucket", "unidentified")

        color = {
            "major": "#fee2e2",
            "minor": "#fef9c3",
            "modified": "#e0f2fe",
            "new": "#dcfce7",
            "deleted": "#fce7f3",
            "nochange": "#ecfeff"
        }.get(bucket, "#f3f4f6")

        out.append(
            f'<tr data-ctype="{bucket}" style="background:{color};">'
            f'<td><a href="#{html_escape(r["req_id"])}" style="text-decoration:underline;color:#2563eb;">'
            f'{html_escape(r["requirement"])}</a></td>'
            f'<td>{html_escape(r["versions"])}</td>'
            f'<td><strong>{html_escape(display_ct)}</strong></td>'
            f'<td>{html_escape(r["description"])}</td>'
            '</tr>'
        )

    out.append('</tbody></table>')

    # JS filter
    out.append("""
<script>
function filterSummary(type) {
  document.querySelectorAll('#summary-table tbody tr').forEach(r => {
    r.style.display = (type === 'all' || r.dataset.ctype === type) ? '' : 'none';
  });
}
</script>
""")

    out.append('</div>')
    # return "\\n".join(out)
    return "\n".join(out)

# -----------------------
# Final HTML generator
# -----------------------
def generate_html_report(
    artifact: Union[str, Dict[str, Any]],
    out_html_path: str = "artifacts/final_validated_change_trace.html",
    title: str = "Validated Change Trace",
    include_diff: bool = False,
    color_changes: bool = False,
    save_json_artifact: bool = True,
    show_images: bool = True,
    show_text: bool = False,
    show_table_body: bool = False,   
) -> str:
    """
    Render validated change trace to an HTML report (multi-version).
    - Reads versions from req["versions"] (falls back to top-level).
    - If kind == "table": do NOT display table body (table suppressed), but show image if available.
    - Else: shows text (and later, image when enabled).
    - Displays change summaries per pair (v1-v2, v2-v3, ...).
    """
    if isinstance(artifact, str):
        with open(artifact, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = artifact

    os.makedirs(os.path.dirname(out_html_path) or ".", exist_ok=True)
    json_out_path = os.path.join(os.path.dirname(out_html_path) or ".", "change_trace.json")
    try:
        if save_json_artifact and isinstance(data, dict):
            with open(json_out_path, "w", encoding="utf-8") as jf:
                json.dump(data, jf, indent=2, ensure_ascii=False)            
    except Exception:
        pass

    reqs = data.get("requirements", []) or []

    parts: List[str] = []
    overall_summary: List[Dict[str, Any]] = []
    # CSS
    ins_style = "background:#eaffea;color:#155724;padding:0 2px;border-radius:3px;font-weight:600" if color_changes else "background:transparent;color:inherit;padding:0 0;border-radius:0;font-weight:inherit"
    del_style = "background:#ffecec;color:#8b0000;padding:0 2px;border-radius:3px;text-decoration:line-through" if color_changes else "background:transparent;color:inherit;padding:0 0;border-radius:0;text-decoration:line-through"

    parts.append(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html_escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{ --fg:#1f2937; --muted:#6b7280; --bg:#fff; --card:#f8fafc; --border:#e5e7eb; }}
  html,body{{margin:0;padding:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}}
  .wrap{{max-width:1200px;margin:24px auto;padding:16px}}
  header.row{{display:flex;align-items:center;gap:12px;margin-bottom:12px}}
  header h1{{font-size:22px;margin:0}}
  .req{{border:1px solid var(--border);border-radius:10px;background:var(--card);margin:18px 0;padding:0;overflow:hidden}}
  .req-hdr{{padding:12px 16px;border-bottom:1px solid var(--border);background:#fff}}
  .req-title{{margin:0;font-size:16px}}
  .pair{{padding:16px;border-top:1px dashed var(--border)}}
  .pair:first-of-type{{border-top:none}}
  .pair-h{{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:baseline;margin-bottom:12px}}
  .badge{{padding:4px 8px;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:12px;font-weight:600}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start}}
  @media (max-width:900px){{.grid{{grid-template-columns:1fr}}}}
  .cell{{border:1px solid var(--border);background:#fff;border-radius:8px;overflow:hidden}}
  .cell-h{{font-size:12px;color:var(--muted);padding:6px 10px;border-bottom:1px solid var(--border);background:#fdfdfd}}
  .cell-body{{padding:10px}}
  .cell-pre{{margin:0 0 8px 0;padding:0;font-family:inherit;white-space:pre-wrap;word-wrap:break-word}}
  .table-html{{overflow:auto;margin-bottom:8px;border:1px solid #eee;padding:6px;border-radius:6px;background:#fff}}
  .table-html table{{border-collapse:collapse;width:100%}}
  .table-html th,.table-html td{{border:1px solid #ddd;padding:6px;text-align:left}}
  .thumb-wrap{{display:flex;gap:8px;align-items:flex-start;flex-wrap:wrap;margin-bottom:8px}}
  .thumb{{max-width:100%;height:auto;border:1px solid var(--border);border-radius:6px;display:block;box-shadow:0 1px 2px rgba(0,0,0,0.04)}}
  .thumb-small{{max-width:260px; width:100%; object-fit:contain}}
  .caption{{font-size:12px;color:var(--muted);margin-top:6px}}
  .diff{{margin-top:10px;padding:10px 12px;border-left:3px solid var(--border);background:#fff;border-radius:4px;font-size:14px}}
  .ins{{{ins_style}}}
  .del{{{del_style}}}
  .hr{{height:1px;background:var(--border);margin:24px 0}}
  .cdesc ul {{ margin:6px 0 6px 18px; padding:0; }}
  .cdesc li {{ margin-bottom:6px; line-height:1.4; }}
  .cdesc .desc-text {{ font-style: italic; color: #374151; margin:6px 0 8px 0; }}
  .adld h4 {{ margin:10px 0 6px 0; font-size:14px; color:#111827; }}
  .adld table {{ border-collapse: collapse; width: 100%; margin:6px 0 8px 0; }}
  .adld th, .adld td {{ border: 1px solid #ddd; padding: 6px; text-align: left; font-size: 13px; }}
</style>
</head>
<body>
<div class="wrap">
  <header class="row">
    <h1>{html_escape(title)}</h1>
    <a href="#overall-summary"
       style="margin-left:auto;font-size:14px;color:#2563eb;text-decoration:underline;">
       Jump to Overall Summary ↓
    </a>
  </header>
""")

    def _preview_text(req_entry: Dict[str, Any]) -> str:
        # Pull the first version's text for the title preview when available
        vdict = req_entry.get("versions") if isinstance(req_entry, dict) and isinstance(req_entry.get("versions"), dict) else req_entry
        vkeys_sorted = _get_version_keys_from_dict(vdict)
        if vkeys_sorted:
            v1k = vkeys_sorted[0]
            val = vdict.get(v1k, "")
            if isinstance(val, dict):
                text_val = val.get("text") or ""
            else:
                text_val = val or ""
            snippet = _normalize_whitespace(str(text_val or ""))
            snippet = snippet[:120] + ("…" if len(snippet) > 120 else "")
            return snippet or "(no preview)"
        return "(no preview)"

    for idx, req in enumerate(reqs, start=1):
        req_kind = (req.get("kind") or "").strip().lower()
        vdict = req.get("versions") if isinstance(req, dict) and isinstance(req.get("versions"), dict) else req
        vkeys_sorted = _get_version_keys_from_dict(vdict)

        preview = _preview_text(req)

        req_anchor = f"req-{idx}"
        parts.append(f'<div class="req" id="{req_anchor}">')
        parts.append(f'  <div class="req-hdr"><h3 class="req-title">Requirement {idx}: {html_escape(preview)}</h3></div>')

        change_log = req.get("Change_log", {}) or {}
        row_level_map = req.get("Row_level_changes", {}) or {}

        # collect pairs (prefer explicit from Change_log)
        render_pairs = _version_pairs_from_change_log(change_log)
        if not render_pairs:
            # fallback: adjacent pairs from versions order
            if len(vkeys_sorted) >= 2:
                render_pairs = [(vkeys_sorted[i], vkeys_sorted[i+1]) for i in range(len(vkeys_sorted)-1)]
            elif vkeys_sorted:
                render_pairs = [(vkeys_sorted[0], vkeys_sorted[0])]

        for (v_from, v_to) in render_pairs:
            a_val = vdict.get(v_from) if isinstance(vdict, dict) else ""
            b_val = vdict.get(v_to) if isinstance(vdict, dict) else ""
            a_text, a_img, a_table_html = _extract_text_img_table(a_val)
            b_text, b_img, b_table_html = _extract_text_img_table(b_val)

            pair_id = f"{v_from}-{v_to}"
            entry = (change_log.get(pair_id) if isinstance(change_log, dict) else None) or {}

            # Prefer Row_level_changes as the authoritative row list
            per_row_list = row_level_map.get(pair_id, []) if isinstance(row_level_map, dict) else []
            if not per_row_list and isinstance(entry, dict) and isinstance(entry.get("pairs"), list):
                per_row_list = entry["pairs"]

            # Merge added/deleted rows from per_row_list (and fall back to entry)
            added_rows: List[Dict[str, Any]] = []
            deleted_rows: List[Dict[str, Any]] = []
            for itm in (per_row_list or []):
                if isinstance(itm, dict):
                    if isinstance(itm.get("added_rows"), list):
                        added_rows.extend(itm["added_rows"])
                    if isinstance(itm.get("deleted_rows"), list):
                        deleted_rows.extend(itm["deleted_rows"])
            if not added_rows and isinstance(entry, dict) and isinstance(entry.get("added_rows"), list):
                added_rows = list(entry["added_rows"])
            if not deleted_rows and isinstance(entry, dict) and isinstance(entry.get("deleted_rows"), list):
                deleted_rows = list(entry["deleted_rows"])

            def _dedup_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                seen = set()
                out = []
                for r in rows or []:
                    if not isinstance(r, dict):
                        continue
                    rk = r.get("row_key", "")
                    rv = tuple(r.get("row_values", [])) if isinstance(r.get("row_values"), list) else ()
                    key = (rk, rv)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(r)
                return out
            added_rows = _dedup_rows(added_rows)
            deleted_rows = _dedup_rows(deleted_rows)            
            ctype_norm = "unidentified"
            cdesc = ""

            # -------- Extract WHOLE-level change correctly --------
            whole_item = None           

            # Fallback to Change_log → pairs
            if not whole_item and isinstance(entry, dict):
                pairs = entry.get("pairs", [])
                if isinstance(pairs, list) and pairs:
                    for p in pairs:
                        if isinstance(p, dict) and "WHOLE" in str(p.get("pair_key", "")):
                            whole_item = p
                            break                 

            # if whole_item:
            #     # AUTHORITATIVE — DO NOT TOUCH AFTER THIS
            #     if whole_item.get("formatting_only") is True:
            #         ctype_norm = "nochange"
            #     else:
            #         ctype_norm = _normalize_change_type(whole_item.get("change_type"))

            #     cdesc = (
            #         whole_item.get("change_description")
            #         or whole_item.get("description")
            #         or "—"
            #     )

            #     overall_summary.append({
            #         "requirement": f"Requirement {idx}",
            #         "req_id": f"req-{idx}",
            #         "versions": f"{v_from}-{v_to}",
            #         "change_type": ctype_norm,
            #         "description": cdesc
            #     })
                       
            ### Reading JSON change_type
            raw_ct = whole_item.get("change_type") or "Unidentified"
            ctype_norm = raw_ct.strip()

            cdesc = (
                whole_item.get("change_description")
                or whole_item.get("description")
                or "—"
            )

            # --- new bucket normalization for tab filtering ---
            bucket_key = ctype_norm.lower().replace(" ", "")

            BUCKET_MAP = {
                "major": "major",
                "majorchange": "major",
                "minor": "minor",
                "minorchange": "minor",
                "nochange": "nochange",
                "unchanged": "nochange",
                "deleted": "deleted",
                "removed": "deleted",
                "new": "new",
                "added": "new",
                "modified": "modified"
            }

            tab_key = BUCKET_MAP.get(bucket_key, "unidentified")

            overall_summary.append({
                "requirement": f"Requirement {idx}",
                "req_id": f"req-{idx}",
                "versions": f"{v_from}-{v_to}",
                "change_type": ctype_norm,   # display 그대로
                "bucket": tab_key,           # tab uses this
                "description": cdesc
            })

            if whole_item is None:
                if added_rows or deleted_rows:
                    ctype_norm = "major"
                    if not cdesc:
                        cdesc = "Rows added and/or deleted."
                elif per_row_list:
                    agg = _aggregate_ct_from_rows(per_row_list)
                    if agg and agg != "unidentified":
                        ctype_norm = agg
                if ctype_norm in ("", "unidentified"):
                    if _content_differs(a_text, a_table_html, b_text, b_table_html):
                        ctype_norm = "modified"
                        if not cdesc:
                            cdesc = "Content changed."      
                            
            # resolve images (disabled unless show_images=True)
            resolved_a_img = _ensure_img_available(a_img, out_html_path) if (show_images and a_img) else None
            resolved_b_img = _ensure_img_available(b_img, out_html_path) if (show_images and b_img) else None

            # --- Render pair UI ---
            parts.append('<div class="pair">')
            parts.append('  <div class="pair-h">')
            parts.append(f'    <span class="badge">{html_escape(v_from)} → {html_escape(v_to)}</span>')
            if ctype_norm:
                parts.append(f'    <span class="ctype">{html_escape(str(ctype_norm).title())}</span>')
            if cdesc:
                parts.append(f'    <div class="cdesc">— {html_escape(str(cdesc))}</div>')
            parts.append('  </div>')  # pair-h

            # Grid with Version A/B — TABLE kind: show image (if available) for tables; otherwise show text (and tables if present)
            parts.append('<div class="grid">')

            # Version A
            parts.append('<div class="cell">')
            parts.append('<div class="cell-h">Version A</div>')
            parts.append('<div class="cell-body">')
            if req_kind == "table":
                # For table kind we DO NOT display the table body; instead display available image (if any)
                if resolved_a_img:
                    parts.append(f'<div class="thumb-wrap"><a href="{html_escape(resolved_a_img)}" target="_blank" rel="noopener noreferrer"><img class="thumb thumb-small" src="{html_escape(resolved_a_img)}" /></a></div>')
                else:
                    parts.append('<div class="cell-pre" style="color:#6b7280">(no image)</div>')
            else:
                # Optionally image (disabled default)
                if resolved_a_img:
                    parts.append(f'<div class="thumb-wrap"><a href="{html_escape(resolved_a_img)}" target="_blank" rel="noopener noreferrer"><img class="thumb thumb-small" src="{html_escape(resolved_a_img)}" /></a></div>')
                # Text (if present) - controlled by show_text
                if show_text and (a_text or "").strip():
                    parts.append(f'<div class="cell-pre">{html_escape(a_text)}</div>')
                # If a table is present for non-table kinds, also show it
                if show_table_body and a_table_html:
                    parts.append(f'<div class="table-html">{a_table_html}</div>')
                #if not ( (show_text and (a_text or "").strip()) or a_table_html or resolved_a_img):
                if not ( (show_text and (a_text or "").strip()) or (show_table_body and a_table_html) or resolved_a_img):
                    parts.append('<div class="cell-pre" style="color:#6b7280">(empty)</div>')
            parts.append('</div></div>')  # A

            # Version B
            parts.append('<div class="cell">')
            parts.append('<div class="cell-h">Version B</div>')
            parts.append('<div class="cell-body">')
            if req_kind == "table":
                # For table kind we DO NOT display the table body; instead display available image (if any)
                if resolved_b_img:
                    parts.append(f'<div class="thumb-wrap"><a href="{html_escape(resolved_b_img)}" target="_blank" rel="noopener noreferrer"><img class="thumb thumb-small" src="{html_escape(resolved_b_img)}" /></a></div>')
                else:
                    parts.append('<div class="cell-pre" style="color:#6b7280">(no image)</div>')
            else:
                if resolved_b_img:
                    parts.append(f'<div class="thumb-wrap"><a href="{html_escape(resolved_b_img)}" target="_blank" rel="noopener noreferrer"><img class="thumb thumb-small" src="{html_escape(resolved_b_img)}" /></a></div>')
                if show_text and (b_text or "").strip():
                    parts.append(f'<div class="cell-pre">{html_escape(b_text)}</div>')
                if show_table_body and b_table_html:
                    parts.append(f'<div class="table-html">{b_table_html}</div>')
                #if not ( (show_text and (b_text or "").strip()) or b_table_html or resolved_b_img):
                if not ( (show_text and (b_text or "").strip()) or (show_table_body and b_table_html) or resolved_b_img):
                    parts.append('<div class="cell-pre" style="color:#6b7280">(empty)</div>')
            parts.append('</div></div>')  # B

            parts.append('</div>')  # grid

            # Inline diff: ONLY for non-table kinds and when both sides are plain text
            if (req_kind != "table") and include_diff and show_text and not (a_table_html or b_table_html):
                diff_html = _inline_diff_html(a_text, b_text)
                if diff_html.strip():
                    parts.append('<div class="diff">' + diff_html + '</div>')

            # ---- Row-level changes block ----
            def _render_cell_changes(cell_changes: List[Dict[str, Any]]) -> str:
                if not cell_changes:
                    return ""
                out = ['<ul class="cell-changes">']
                for c in cell_changes:
                    col_hdr = c.get("col_header") or (str(c.get("col_index")) if c.get("col_index") is not None else "")
                    oldv = c.get("old", "")
                    newv = c.get("new", "")
                    out.append(
                        f'<li><strong>{html_escape(str(col_hdr))}:</strong> '
                        f'{html_escape(str(oldv))} → {html_escape(str(newv))}</li>'
                    )
                out.append("</ul>")
                return "\n".join(out)

            per_row_list = row_level_map.get(pair_id, []) if isinstance(row_level_map, dict) else []
            if not per_row_list and isinstance(entry, dict) and isinstance(entry.get("pairs"), list):
                per_row_list = entry["pairs"]

            if isinstance(per_row_list, list) and per_row_list:
                parts.append('<div class="cdesc">')
                parts.append('<strong>Row-level changes:</strong>')
                parts.append('<ul>')
                for item in per_row_list:
                    if not isinstance(item, dict):
                        continue
                    row_key = item.get("row_key") or item.get("pair_key") or ""
                    row_label = _readable_row_label_from_pair_key(str(row_key))
                    rct = _normalize_change_type(item.get("change_type"))
                    rdesc = item.get("change_description") or item.get("description") or ""
                    line = f'<li><strong>{html_escape(row_label or "Row")}</strong> — {html_escape(rct.title())}'
                    if rdesc:
                        line += f': {html_escape(str(rdesc))}'
                    line += '</li>'
                    parts.append(line)

                    # show cell_changes list if present
                    cell_changes = item.get("cell_changes") or []
                    # also handle nested row_changes (WHOLE -> row list)
                    if not cell_changes and isinstance(item.get("row_changes"), list) and item["row_changes"]:
                        for sub in item["row_changes"]:
                            if isinstance(sub, dict):
                                cc = sub.get("cell_changes") or []
                                if cc:
                                    parts.append(_render_cell_changes(cc))
                    else:
                        parts.append(_render_cell_changes(cell_changes))
                parts.append('</ul>')
                parts.append('</div>')

            # ---- Added/Deleted rows (deduped) ----
            def _render_adld_rows(rows: List[Dict[str, Any]], label: str) -> str:
                if not rows:
                    return ""
                headers = []
                first = rows[0] if rows else None
                if isinstance(first, dict) and isinstance(first.get("row_values"), list):
                    n = len(first["row_values"])
                    headers = ["col" + str(i) for i in range(n)]
                out = [f'<div class="adld"><h4>{html_escape(label)}</h4><table><thead><tr><th>row_key</th>']
                for h in headers:
                    out.append(f'<th>{html_escape(h)}</th>')
                out.append('</tr></thead><tbody>')
                for r in rows:
                    rk = r.get("row_key", "")
                    rv = r.get("row_values", [])
                    out.append(f'<tr><td>{html_escape(str(rk))}</td>')
                    for val in rv:
                        out.append(f'<td>{html_escape(str(val))}</td>')
                    out.append('</tr>')
                out.append('</tbody></table></div>')
                return "\n".join(out)

            added_rows = added_rows if 'added_rows' in locals() else []
            deleted_rows = deleted_rows if 'deleted_rows' in locals() else []

            if added_rows:
                parts.append(_render_adld_rows(added_rows, "Added rows"))
            if deleted_rows:
                parts.append(_render_adld_rows(deleted_rows, "Deleted rows"))

            parts.append('</div>')            

        parts.append('</div>')  # .req
        parts.append('<div class="hr"></div>')

    #summary_html = _render_overall_summary_table(overall_summary)
    summary_html = _render_overall_summary_table(overall_summary)
    if summary_html:
        parts.append('<div class="hr"></div>')
        parts.append(summary_html)


    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    return out_html_path