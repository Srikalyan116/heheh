import os
import json
import re
import mlflow
from typing import Dict, Any, List, Tuple, Union
from difflib import SequenceMatcher

def _is_markdown_table(text: str) -> bool:
    if not isinstance(text, str):
        return False
    # Simple heuristic: looks like a GFM table
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return False
    return lines[0].startswith("|") and ("|" in lines[0]) and ("---" in lines[1])

def _normalize_whitespace(s: str) -> str:
    s = "" if s is None else str(s)
    return " ".join(s.split())

def _word_tokens(s: str) -> List[str]:
    # Tokenize on words + punctuation so we can diff better
    return re.findall(r"\w+|[^\w\s]", s, re.UNICODE)

def _word_level_diff(a: str, b: str) -> str:
    """
    Return a compact inline diff:
      - insertions in **bold**
      - deletions in ~~strikethrough~~
    """
    a_tokens = _word_tokens(a)
    b_tokens = _word_tokens(b)

    sm = SequenceMatcher(None, a_tokens, b_tokens)
    out: List[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.extend(a_tokens[i1:i2])
        elif tag == "insert":
            out.append("**" + " ".join(b_tokens[j1:j2]) + "**")
        elif tag == "delete":
            out.append("~~" + " ".join(a_tokens[i1:i2]) + "~~")
        elif tag == "replace":
            out.append("~~" + " ".join(a_tokens[i1:i2]) + "~~")
            out.append("**" + " ".join(b_tokens[j1:j2]) + "**")
    # squash extra spaces
    s = " ".join(out)
    s = re.sub(r"\s+\.\s+", ". ", s)
    s = re.sub(r"\s+,", ",", s)
    return s.strip()

def _block(s: str) -> str:
    """
    Wrap content into a <pre> block to preserve tables and layout inside a Markdown table cell.
    """
    s = "" if s is None else str(s)
    return f"<pre>{s}</pre>"

def _version_pairs(req: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Return adjacent version pairs present in this requirement, e.g. [("V1","V2"), ("V2","V3"), ...]
    """
    version_keys = [k for k in req.keys() if k.lower().startswith("v")]
    # sort by numeric tail: v1, v2, v3...
    version_keys = sorted(version_keys, key=lambda x: int(re.findall(r"\d+", x)[0]))
    pairs = []
    for i in range(len(version_keys) - 1):
        pairs.append((version_keys[i], version_keys[i+1]))
    return pairs

def generate_markdown_report(
    artifact: Union[str, Dict[str, Any]],
    out_md_path: str = "artifacts/final_validated_change_trace.md",
    title: str = "Validated Change Trace (Markdown View)",
    include_diff: bool = True
) -> str:
    """
    Convert (validated) change trace JSON (dict or path) to a readable Markdown report
    with side-by-side version content and a compact word-level diff.
    """
    if isinstance(artifact, str):
        with open(artifact, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = artifact

    reqs = data.get("requirements", [])
    lines: List[str] = []
    lines.append(f"# {title}\n")
    token_usage = data.get("token_usage")
    if token_usage:
        lines.append("**Token usage:** " + ", ".join([f"{k}: {v}" for k, v in token_usage.items()]) + "\n")

    for idx, req in enumerate(reqs, start=1):
        # make a short heading preview from the earliest version's text
        first_v = sorted([k for k in req.keys() if k.lower().startswith("v")], key=lambda x: int(re.findall(r"\d+", x)[0]))[0]
        preview = _normalize_whitespace(req.get(first_v.lower(), ""))[:120]
        if len(preview) == 120:
            preview += "…"

        lines.append(f"## Requirement {idx}: {preview}\n")

        # Render each pair present in Change_log, falling back to adjacency if needed
        change_log = req.get("Change_log", {}) or {}
        pairs = _version_pairs(req)
        pairs_from_log = []
        for pair_id in change_log.keys():
            if "-" in pair_id:
                pairs_from_log.append(tuple(pair_id.split("-")))
        # prefer listed pairs in Change_log order, else fall back to adjacency
        render_pairs = pairs_from_log if pairs_from_log else pairs

        for pair in render_pairs:
            v_from, v_to = pair
            a_txt = req.get(v_from.lower(), "") or ""
            b_txt = req.get(v_to.lower(), "") or ""
            entry = change_log.get(f"{v_from}-{v_to}", {})

            ctype = entry.get("change_type", "Unidentified")
            cdesc = entry.get("change_description", "")

            lines.append(f"### {v_from} → {v_to} — **{ctype}**")
            if cdesc:
                lines.append(f"> {cdesc}\n")

            # Side-by-side presentation as a Markdown table with <pre> blocks
            lines.append("| Version A | Version B |")
            lines.append("|---|---|")
            lines.append(f"| {_block(a_txt)} | {_block(b_txt)} |")
            lines.append("")

            if include_diff:
                # Word-level diff (one line)
                diff_line = _word_level_diff(a_txt, b_txt)
                if diff_line:
                    lines.append("**Inline diff:**")
                    lines.append(diff_line + "\n")

        # spacer
        lines.append("---\n")

    # ensure folder exists then write
    os.makedirs(os.path.dirname(out_md_path), exist_ok=True)
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # optionally log to MLflow if an active run exists
    try:
        mlflow.log_artifact(out_md_path, artifact_path=os.path.dirname(out_md_path))
    except Exception:
        pass

    return out_md_path
