# llm_judge.py
import os
import re
import json
import mlflow
import asyncio
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from app.delta_comparator.utils.logger import log as logging
from openai import AsyncAzureOpenAI  # you said you use AsyncAzureOpenAI
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# --- Configuration / Rules load ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_PATH = os.path.join(BASE_DIR, "rules.json")
with open(RULES_PATH, "r", encoding="utf-8") as f:
    RULES = json.load(f)

MODEL_NAME = os.getenv("AZURE_MODEL")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")


# Async client
client = AsyncAzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version="2024-05-01-preview",
    azure_endpoint=AZURE_ENDPOINT,
)

# Trivial labels that don't need LLM judge
TRIVIAL_TYPES = {"no change", "skipped", "new", "deleted"}

# ---------------- helpers ----------------
def extract_json(text: str) -> Dict[str, Any]:
    """Extract first JSON object from a model response string."""
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in text")
        return json.loads(m.group())
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from LLM output: {e}")


def normalize_label(label: Optional[str]) -> str:
    if not isinstance(label, str):
        return ""
    label = label.strip().lower().replace("\u00A0", " ")
    if label in ("no change", "no changes"):
        return "no change"
    if "minor" in label:
        return "minor"
    if "major" in label:
        return "major"
    if label in ("skipped",):
        return "skipped"
    if label in ("new",):
        return "new"
    if label in ("deleted",):
        return "deleted"
    return label


def build_llm_judge_prompt(original: str, revised: str, pred: str, rules: Dict) -> str:
    """Construct judge prompt. Keep short and deterministic (temperature=0)."""
    return f"""
You are an evaluation LLM. Judge whether the predicted change type is correct given Original and Revised.

Rules (use these only):
MAJOR RULES:
- {rules['major_rules'][0] if rules.get('major_rules') else ''}
- {rules['major_rules'][1] if rules.get('major_rules') and len(rules['major_rules'])>1 else ''}
MINOR RULES:
- {rules['minor_rules'][0] if rules.get('minor_rules') else ''}
NO CHANGE RULES:
- {rules['no_change_rules'][0] if rules.get('no_change_rules') else ''}

Original:
{original}

Revised:
{revised}

Predicted change type (from tracer):
{pred}

Return STRICT JSON only with fields:
{{ "change_type": "Major | Minor | No Change | Skipped | New | Deleted", 
   "score": 0.0,               # 1.0 exact / 0..1 float
   "reason": "short explanation",
   "confidence": 0.0           # optional confidence for reasoning (0..1)
}}
"""  # note: model may return other fields; parser is tolerant


# ---------------- LLM call (async) ----------------
async def call_llm_as_judge(prompt: str) -> Dict[str, Any]:
    """
    Async call to Azure LLM judge.
    Returns a dict with keys: change_type, score, reason, confidence, total_tokens
    Always returns total_tokens (int) even on failure (0).
    """
    try:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
        )

        raw = (resp.choices[0].message.content or "").strip()
        #logging.debug(f"Raw Judge Response: {raw}")

        # parse JSON block
        obj = {}
        try:
            obj = extract_json(raw)
        except Exception as e:
            # include raw for debugging but return a graceful failure
            return {
                "change_type": "Unidentified",
                "score": -1.0,
                "reason": f"Failed to parse JSON from model output: {e}. Raw start: {raw[:400]}",
                "confidence": 0.0,
                "total_tokens": _extract_usage_total(resp),
            }

        # Normalize fields with safe defaults
        change_type = obj.get("change_type") or obj.get("type") or obj.get("classification") or "Unidentified"
        score = float(obj.get("score", obj.get("score_value", 0.0) or 0.0))
        reason = str(obj.get("reason", obj.get("explanation", "") or "")).strip()
        confidence = float(obj.get("confidence", obj.get("confidence_score_for_reasoning", 0.0) or 0.0))

        # token usage extraction (best-effort)
        total_tokens = _extract_usage_total(resp)

        # log per-call tokens as metrics (optional — increases metric cardinality)
        try:
            mlflow.log_metric("judge_call_total_tokens", int(total_tokens))
        except Exception:
            pass

        return {
            "change_type": change_type,
            "score": score,
            "reason": reason,
            "confidence": confidence,
            "total_tokens": int(total_tokens),
        }

    except Exception as e:
        # model call failed
        return {
            "change_type": "Unidentified",
            "score": -1.0,
            "reason": f"LLM call error: {e}",
            "confidence": 0.0,
            "total_tokens": 0,
        }


def _extract_usage_total(resp) -> int:
    """
    Robust extraction of token usage from response. The 'usage' might be an object
    with attributes or a dict. Return 0 if not available.
    """
    try:
        usage = getattr(resp, "usage", None)
        if usage is None and isinstance(resp, dict):
            usage = resp.get("usage")
        if usage is None:
            return 0
        # If object with attributes
        total = getattr(usage, "total_tokens", None)
        if total is not None:
            return int(total)
        # If dict-like
        if isinstance(usage, dict):
            return int(usage.get("total_tokens", 0) or 0)
        # last resort: try indexing
        try:
            return int(usage["total_tokens"])
        except Exception:
            return 0
    except Exception:
        return 0

# ---------------- Main evaluate function ----------------
async def evaluate_artifact_with_llm(
    artifact: dict,
    rules: dict,
    task_id: str,
    processed_folder: str,
    max_concurrency: int = 6,
) -> Tuple[dict, int]:
    """
    Evaluate artifact (change_trace-like dict).
    Returns (eval_result_dict, total_judge_tokens).
    - trivial predictions are auto-accepted with score=1
    - calls LLM only for predicted major/minor
    - supports dynamic pairs: V1-V2, V2-V3, ...
    """
    requirements = artifact.get("requirements", []) or []
    results: List[Dict[str, Any]] = []
    
    # Aggregates
    total_major_predicted = total_minor_predicted = 0
    total_major_judge = total_minor_judge = 0
    total_judge_tokens = 0

    sem = asyncio.Semaphore(max_concurrency)

    async def process_pair(req: Dict[str, Any], pair_id: str, change: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal total_major_predicted, total_minor_predicted, total_major_judge, total_minor_judge, total_judge_tokens

        pred = normalize_label(change.get("change_type", ""))
        # determine v_from and v_to robustly
        try:
            v_from, v_to = pair_id.split("-")
        except Exception:
            # fallback: find keys in req that start with 'v' (V1,V2,..) and map by index
            # If can't parse, use empty strings
            v_from, v_to = "v1", "v2"

        original = str(req.get(v_from.lower(), "") or "")
        revised = str(req.get(v_to.lower(), "") or "")

        # Trivial auto-accept (no LLM call)
        if pred in TRIVIAL_TYPES or pred == "":
            judge_type_title = {
                "no change": "No Change",
                "skipped": "Skipped",
                "new": "New",
                "deleted": "Deleted",
                "": "No Change"
            }.get(pred, "No Change")
            return {
                "pair_id": pair_id,
                "v_from": original,
                "v_to": revised,
                "predicted": pred,
                "judge_change_type": judge_type_title,
                "judge_predicted": "correct",
                "judge_score": 1.0,
                "judge_reason": "Trivial change, auto-accepted",
                "confidence_score_for_reasoning": 1.0,
            }

        # Track predicted counts
        if pred == "major":
            total_major_predicted += 1
        elif pred == "minor":
            total_minor_predicted += 1

        # Non-trivial -> call LLM judge (bounded concurrency)
        async with sem:
            prompt = build_llm_judge_prompt(original, revised, pred, rules)
            judged = await call_llm_as_judge(prompt)

        # accumulate tokens
        used = int(judged.get("total_tokens", 0) or 0)
        total_judge_tokens += used

        judge_type_norm = normalize_label(judged.get("change_type", "Unidentified"))
        judge_type_title = {
            "major": "Major",
            "minor": "Minor",
            "no change": "No Change",
            "skipped": "Skipped",
            "new": "New",
            "deleted": "Deleted"
        }.get(judge_type_norm, "Unidentified")

        is_correct = (judge_type_norm == pred)

        if judge_type_norm == "major" and is_correct:
            total_major_judge += 1
        if judge_type_norm == "minor" and is_correct:
            total_minor_judge += 1

        return {
            "pair_id": pair_id,
            "v_from": original,
            "v_to": revised,
            "predicted": pred,
            "judge_change_type": judge_type_title,
            "judge_predicted": "correct" if is_correct else "incorrect",
            "judge_score": float(judged.get("score", 0.0) or 0.0),
            "judge_reason": judged.get("reason", ""),
            "confidence_score_for_reasoning": float(judged.get("confidence", 0.0) or 0.0),
        }

    # create tasks for all pairs
    tasks = []
    for req in requirements:
        change_log = req.get("Change_log", {}) or {}
        for pair_id, change in change_log.items():
            tasks.append(asyncio.create_task(process_pair(req, pair_id, change)))

    # run them
    rows = []
    if tasks:
        rows = await asyncio.gather(*tasks)
    results.extend(rows)

    # compute aggregated metrics
    valid_scores = [r["judge_score"] for r in results if isinstance(r.get("judge_score"), (int, float)) and r["judge_score"] >= 0]
    avg_judge_score = float(sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0
    failure_rate = float(sum(1 for s in valid_scores if s <= 0.5) / len(valid_scores)) if valid_scores else 0.0
    score_distribution = {
        "perfect": int(sum(1 for s in valid_scores if s == 1.0)),
        "medium": int(sum(1 for s in valid_scores if 0.5 < s < 1.0)),
        "fail": int(sum(1 for s in valid_scores if s <= 0.5)),
    }

    # mlflow logging (nested run recommended from caller)
    try:
        mlflow.log_metric("avg_judge_score", avg_judge_score)
        mlflow.log_metric("failure_rate", failure_rate)
        mlflow.log_metric("total_major_predicted", total_major_predicted)
        mlflow.log_metric("total_minor_predicted", total_minor_predicted)
        mlflow.log_metric("total_major_judge", total_major_judge)
        mlflow.log_metric("total_minor_judge", total_minor_judge)
        mlflow.log_metric("total_judge_tokens", int(total_judge_tokens))
    except Exception:
        pass

    eval_result = {
        "task_id": task_id,
        "evaluation_results": results,
        "num_requirements": len(requirements),
        "avg_judge_score": avg_judge_score,
        "failure_rate": failure_rate,
        "score_distribution": score_distribution,
        "total_major_predicted": total_major_predicted,
        "total_minor_predicted": total_minor_predicted,
        "total_major_judge": total_major_judge,
        "total_minor_judge": total_minor_judge,
    }

    # Save JSON + CSV/XLSX artifacts for preview & upload
    os.makedirs(processed_folder, exist_ok=True)
    json_path = os.path.join(processed_folder, f"{task_id}_llm_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)
    # try:
    #     mlflow.log_artifact(json_path, artifact_path="llm_eval")
    # except Exception:
    #     pass

    # Save CSV for MLflow preview
    try:
        df = pd.DataFrame(results, columns=[
            "pair_id", "v_from", "v_to", "predicted",
            "judge_change_type", "judge_predicted",
            "judge_score", "judge_reason", "confidence_score_for_reasoning"
        ])
        csv_path = os.path.join(processed_folder, f"{task_id}_llm_eval.csv")
        df.to_csv(csv_path, index=False)
        # mlflow.log_artifact(csv_path, artifact_path="llm_eval")
        
    except Exception:
        pass
    mlflow.set_tag("Models", MODEL_NAME)
    return eval_result, int(total_judge_tokens)
