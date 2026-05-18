"""Utility file for Notification helper function."""
import os
import json
import pandas as pd
from app.delta_comparator.utils.logger import log as logging
import shutil
from typing import List, Dict, Optional
from dotenv import load_dotenv
import re
from collections import defaultdict
from io import BytesIO
#import mlflow
#import mlflow.tracing.export.async_export_queue as async_export_queue
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time
import os
from app.delta_comparator.core.embedding import Vectorize
from app.delta_comparator.core.query_processor import QueryProcessor   
from app.delta_comparator.core.revision_comparer import RevisionComparer
from app.delta_comparator.core.converter import ConverterProcessor
from app.delta_comparator.core.pdf_extractor import PDFExtractor
from app.delta_comparator.utils.mongo_db import MongoDB



from pathlib import Path

# DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
# DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
# MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI")
# MLFLOW_EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID")

async def run_tracer_from_mapped(csv_path, run_name, task_id, artifact_dir, batch_size, max_concurrency, RULES):
    """
    Reads the mapped CSV and produces change_trace.json in artifacts/.
    Row order is preserved from the CSV.
    Also logs params/metrics/artifacts to MLflow.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    out_json = os.path.join(artifact_dir, "change_tracer.json")

    t0 = time.time()
    #logging.debug(f"Starting ChangeTracer: csv='{csv_path}', artifact_dir='{artifact_dir}'")
    
    # --- load & normalize ---
    df = pd.read_csv(csv_path).fillna("")
    df["_row_order"] = range(len(df))  # preserve exact original order
    #logging.debug(f"Loaded mapped CSV: {len(df)} rows, {len(df.columns)} columns")

    # optional legacy guards (kept for compatibility)
    for col in [
        "type", "source_text", "revision_text",
        "source_img_path", "revision_img_path",
        "source_table_body", "revision_table_body",
        "source_image_text_extracted", "revision_image_text_extracted",
    ]:
        if col not in df.columns:
            df[col] = ""

    df = df.sort_values("_row_order").reset_index(drop=True)

    # --- MLflow params ---
    # mlflow.log_param("csv_path", os.path.relpath(csv_path))
    # mlflow.log_param("n_rows", len(df))
    # mlflow.log_param("artifact_dir", artifact_dir)
    # mlflow.log_param("batch_size", batch_size)
    # mlflow.log_param("max_concurrency", max_concurrency)
    # mlflow.log_param("rules_major", len(RULES.get("major_rules", [])))
    # mlflow.log_param("rules_minor", len(RULES.get("minor_rules", [])))
    # mlflow.log_param("rules_no_change", len(RULES.get("no_change_rules", [])))
    # mlflow.set_tag("Models", os.getenv("AZURE_MODEL_GPT"))

    # --- run comparer ---
    comparer = RevisionComparer()
    #logging.debug("Invoking RevisionComparer.run_and_log(...)")
    final_json, tokens_used = await comparer.run_and_log(
        df,
        RULES,
        task_id=task_id,
        artifact_path=out_json,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    )

    # --- persist + log artifact/metrics ---
    if os.path.isfile(out_json):
        # mlflow.log_artifact(out_json, artifact_path="artifacts")
        logging.debug(f"Logged artifact")
    else:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)
        # mlflow.log_artifact(out_json, artifact_path="artifacts")
        logging.warning(f"Output file was missing post-run; rewrote: {out_json}")

    elapsed = time.time() - t0
    # mlflow.log_metric("aggregate_total_tokens", int(tokens_used))
    # mlflow.log_metric("tracer_total_time_sec", elapsed)
    # mlflow.set_tag("runner", "run_pipeline_from_mapped.py")

    #logging.debug(f"ChangeTracer completed in {elapsed}s | tokens={tokens_used}")
    #logging.debug(f"Wrote change trace JSON to {out_json}")

    return final_json, tokens_used