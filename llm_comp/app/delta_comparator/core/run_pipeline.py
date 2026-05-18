import asyncio
import json
import os

import mlflow
from datetime import datetime
from typing import Optional, Any, Dict
import re
import time
import shutil
import warnings
from contextlib import suppress
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage

from rabbitMQ_manager import RabbitMQManager, create_queue_publish
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os, time, asyncio, functools, json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from app.delta_comparator.utils.logger import set_trace_context,set_job_context
from app.delta_comparator.core.logging_setup import configure_stdout_logging
configure_stdout_logging("INFO")  # parent logs to stdout

# --- Optional: cap BLAS threads from code (no env/config maps needed) ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from app.delta_comparator.core.pipeline_nonid import run_tracer_from_mapped
from app.delta_comparator.core.llm_judge import evaluate_artifact_with_llm
from app.delta_comparator.core.report_md import generate_markdown_report
from app.delta_comparator.core.report_html import generate_html_report
from app.delta_comparator.utils.constant import *
from app.delta_comparator.utils.helper import get_blob_segment
from app.delta_comparator.utils.azure_helper import BlobClientManager
from app.delta_comparator.utils.mongo_db import MongoDB
from app.delta_comparator.core.type_filtered_runner import create_mappings, create_mappings_async
from app.delta_comparator.utils.logger import log as logging
import aiofiles
#from app.delta_comparator.rabbitMQ_manager import RabbitMQClient

##rabbitmq_sync = RabbitMQClient()
# One shared process pool for CPU/file-heavy work.
# Tweak max_workers if needed.
_PROCESS_POOL = ProcessPoolExecutor(
    max_workers=max(1, (os.cpu_count() or 2) // 2)
)


# DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
# DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
# MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI")
# MLFLOW_EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_PATH = os.path.join(BASE_DIR, "rules.json")

with open(RULES_PATH, "r", encoding="utf-8") as f:
    RULES = json.load(f)

async def transform_json(path, key_to_change):
    # Keep file I/O off the event loop.
    data = await asyncio.to_thread(_read_json_file, path)

    transformed_data = []

    for item in data:
        new_item = {}
        for key, value in item.items():
            
            if key==key_to_change:
                new_item["text"] = value
            else:
                new_item[key] = value
        transformed_data.append(new_item)

    await asyncio.to_thread(_write_json_file, path, transformed_data)


def _read_json_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_file(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _read_binary_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _task_cache_dir(task_id: str) -> str:
    app_folder = os.path.abspath(os.path.dirname(__file__))
    parent_folder = os.path.dirname(app_folder)
    return os.path.join(parent_folder, CACHE, DELTA_COMPARATOR, task_id)


async def _cleanup_task_cache(task_id: str) -> None:
    task_dir = _task_cache_dir(task_id)
    if os.path.isdir(task_dir):
        await asyncio.to_thread(shutil.rmtree, task_dir, True)
        logging.info(f"Cleared task cache directory: {task_dir}")

_POOL = None
def get_pool():
    global _POOL
    if _POOL is not None:
        return _POOL
    try:
        ctx = mp.get_context("spawn")
        _POOL = ProcessPoolExecutor(mp_context=ctx, max_workers=1)
        #logging.debug("Process pool created with 'spawn' (max_workers=1).")
    except Exception as e:
        logging.warning(f"Spawned pool not available ({e}). Falling back to ThreadPool.")
        _POOL = ThreadPoolExecutor(max_workers=1)
    return _POOL

async def run_pipeline(
    revised_paths: list[str],
    type_filter: list[str],
    task_id: str,
    key_column = {},
    rules: dict = RULES,
) -> dict:

    try:
        mongo_client = MongoDB()
        #logging.debug("MongoDB client created")
        app_folder = os.path.abspath(os.path.dirname(__file__))
        #logging.debug(f"App folder : {app_folder}")
        parent_folder = os.path.dirname(app_folder)
        #logging.debug(f"Parent folder : {parent_folder}")
        processed_folder = os.path.join(
            parent_folder, CACHE, DELTA_COMPARATOR, task_id, PROCESSED)
        #logging.debug(f"Processed folder : {processed_folder}")
        os.makedirs(processed_folder, exist_ok=True)
        #logging.debug(f"Path join completed")
        # If you still use MLflow, keep your existing init/start_run around this block.
        run_name = f"Delta_Comparator_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        #logging.debug(f"Starting run: {run_name}")

        loop = asyncio.get_running_loop()
        #logging.debug(f"Loop info: {loop}")
        # pool = get_pool()
        # logging.debug(f"Pool info : {pool}")
        # 1) Heavy mapping offloaded to worker
        t0 = time.time()
        completion_status = {
                    "status": "MAPPINGS STARTED",
                    "task_id": task_id,
                    "status_code": 202,
                    "status_message": f"Mappings Creation Started",
                    "output_file_url_json": None,
                    "output_file_url": None,
                    "collection_name": DELTA_COMPARATOR_COLLECTION,
        }

        await RabbitMQManager().publish_completion(task_id, completion_status)

        logging.info(f"Mapping creation started at {t0},{revised_paths}, {type_filter}, {processed_folder}")
        result = {
                    "type": "stream",
                    "task_id": task_id,
                    "kind": None,
                    "completed_ratio": 11,
                    "message": "Mapping creation started",
                }
        queue_name = f"delta_comparator_{task_id}"
        await create_queue_publish(queue_name, result)
        try:
            await create_mappings_async(revised_paths, type_filter, processed_folder, key_column, task_id)
        except Exception as e:
            logging.error(f"Mapping Creation Failed: {e}")
            completion_status = {
                        "task_id": task_id,
                        "status_code": 500,
                        "status_message": "Mapping Creation Failed",
                        "output_file_url_json": None,
                        "output_file_url": None,
                        "collection_name": DELTA_COMPARATOR_COLLECTION,
                        "status": "FAILED"
            }
            await RabbitMQManager().publish_completion(task_id, completion_status)
            raise e
        #logging.debug(f"Mapping creation finished.")
        result = {
                    "type": "stream",
                    "task_id": task_id,
                    "kind": None,
                    "completed_ratio": 80,
                    "message": "Mapping creation finished",
                }
        queue_name = f"delta_comparator_{task_id}"
        await create_queue_publish(queue_name, result)
        dt = time.time() - t0
        logging.info(f"create_mappings completed in {dt}s")
        mongo_client.update_status_for_task_id(task_id, "MAPPINGS BETWEEN REVISION CREATED","STREAMING")
        completion_status = {
                    "status": "RUNNING CHANGE TRACER",
                    "task_id": task_id,
                    "status_code": 202,
                    "status_message": "Mappings Creation Completed and Change Tracer Started",
                    "output_file_url_json": None,
                    "output_file_url": None,
                    "collection_name": DELTA_COMPARATOR_COLLECTION,
        }
        await RabbitMQManager().publish_completion(task_id, completion_status)
        # 2) Async tracer/LLM stage
        #csv_path = os.path.join(processed_folder, "combined_mapping.csv")
        result = {
                    "type": "stream",
                    "task_id": task_id,
                    "kind": None,
                    "completed_ratio": 80,
                    "message": "Running change tracer on mapped data",
                }
        queue_name = f"delta_comparator_{task_id}"
        await create_queue_publish(queue_name, result)

        tracer_liveness_stop = asyncio.Event()
        tracer_liveness_interval_seconds = 20
        tracer_liveness_start_ratio = 81
        tracer_liveness_end_ratio = 90

        async def _tracer_liveness_heartbeat():
            current_ratio = tracer_liveness_start_ratio
            while not tracer_liveness_stop.is_set():
                await asyncio.sleep(tracer_liveness_interval_seconds)
                if tracer_liveness_stop.is_set():
                    return
                await create_queue_publish(
                    queue_name,
                    {
                        "type": "stream",
                        "task_id": task_id,
                        "kind": "llm",
                        "completed_ratio": current_ratio,
                        "message": "Change tracer still running",
                    },
                )
                if current_ratio < tracer_liveness_end_ratio:
                    current_ratio += 1

        tracer_liveness_task = asyncio.create_task(_tracer_liveness_heartbeat())
        try:
            final_json, tokens_tracer = await run_tracer_from_mapped(
                csv_path=processed_folder + "/combined_mapping.csv",
                run_name="ChangeTracer",
                task_id=task_id,
                artifact_dir=processed_folder + "/artifacts",
                batch_size=6,
                max_concurrency=6,
                RULES=rules,
            )
        except Exception as e:
            logging.error(f"Change Tracer Failed: {e}")
            completion_status = {
                        "task_id": task_id,
                        "status_code": 500,
                        "status_message": "Change Tracer Failed",
                        "output_file_url_json": None,
                        "output_file_url": None,
                        "collection_name": DELTA_COMPARATOR_COLLECTION,
                        "status": "FAILED"
            }
            await RabbitMQManager().publish_completion(task_id, completion_status)
            raise e
        finally:
            tracer_liveness_stop.set()
            tracer_liveness_task.cancel()
            await asyncio.gather(tracer_liveness_task, return_exceptions=True)
        result = {
                    "type": "stream",
                    "task_id": task_id,
                    "kind": None,
                    "completed_ratio": 90,
                    "message": "Change tracer completed",
                }
        queue_name = f"delta_comparator_{task_id}"
        await create_queue_publish(queue_name, result)
        completion_status = {
                    "task_id": task_id,
                    "status_code": 200,
                    "status_message": f"Change tracer Completed for {str(type_filter)}",
                    "output_file_url_json": None,
                    "output_file_url": None,
                    "collection_name": DELTA_COMPARATOR_COLLECTION,
        }
        await RabbitMQManager().publish_completion(task_id, completion_status)
        logging.info(f"run_tracer_from_mapped done. tokens={tokens_tracer}")
        return final_json
    except Exception as e:
        logging.error(f"Error in run_pipeline: {e}")
        mongo_client.update_status_for_task_id(task_id, "FAILED")
        completion_status = {
                    "task_id": task_id,
                    "status_code": 500,
                    "status_message": "Run Pipeline Failed",
                    "output_file_url_json": None,
                    "output_file_url": None,
                    "collection_name": DELTA_COMPARATOR_COLLECTION,
                    "status": "FAILED"
        }
        await RabbitMQManager().publish_completion(task_id, completion_status)
        raise e

# async def run_pipeline(
#         revised_paths: list[str],
#         type_filter: list[str],
#         task_id: str,
#         rules: dict = RULES,
#     ) -> dict:

#     mongo_client = MongoDB()
#     app_folder = os.path.abspath(os.path.dirname(__file__))
#     parent_folder = os.path.dirname(app_folder)
#     processed_folder = os.path.join(
#         parent_folder, CACHE, DELTA_COMPARATOR, task_id, PROCESSED
#     )
#     os.makedirs(processed_folder, exist_ok=True)

#     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#     mlflow.autolog()
#     run_name = f"Delta_Comparator_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

#     final_json = {}

#     loop = asyncio.get_running_loop()

#     with mlflow.start_run(experiment_id=int(os.getenv("MLFLOW_EXPERIMENT_ID")), run_name=run_name):

#         # OFFLOAD the heavy sync mapping step to the process pool
#         t0 = time.time()
#         await loop.run_in_executor(
#             _PROCESS_POOL,
#             functools.partial(
#                 create_mappings,           # <- your sync function
#                 revised_paths,
#                 type_filter,
#                 processed_folder
#             )
#         )
#         t1 = time.time() - t0
#         mlflow.log_metric("Time for map creations", t1)
#         .logging.info(f"Time taken for creating mappings: {t1} seconds")
#         mongo_client.update_status_for_task_id(task_id, "MAPPINGS BETWEEN REVISION CREATED")

#         # Now run the async LLM/tracer step (non-blocking, respects max_concurrency)
#         final_json, tokens_tracer = await run_tracer_from_mapped(
#             csv_path=processed_folder + "/combined_mapping.csv",
#             run_name="ChangeTracer",
#             artifact_dir=processed_folder + "/artifacts",
#             batch_size=6,
#             max_concurrency=6,
#             RULES=rules
#         )

#     return final_json


async def download_raw_files(task_id, revision_filenames, created_time):
    """
    Downloads source and revision files from Azure Blob Storage in parallel.

    Args:
        task_id (str): Unique identifier for the task.
        revision_filenames (list): List of filenames to download.
        created_time (str): Timestamp for blob path construction.

    Returns:
        tuple: (local_source_path, local_target_path, output_dir_path)
    """
    mongo_client = MongoDB()
    app_folder = os.path.abspath(os.path.dirname(__file__))
    parent_folder = os.path.dirname(app_folder)
    tmp_source_pdf_folder = os.path.join(
        parent_folder, CACHE, DELTA_COMPARATOR, task_id, RAW
    )
    temp = get_blob_segment(
        task_id,
        DELTA_COMPARATOR,
        created_time,
    )
    output_dir_path = os.path.join(
        parent_folder, CACHE, DELTA_COMPARATOR, task_id, PROCESSED
    )
    os.makedirs(output_dir_path, exist_ok=True)
    blob_manager = BlobClientManager()
    queue_name = f"delta_comparator_{task_id}"

    async def publish_download_update(message: str, completed_ratio: int):
        result = {
            "type": "stream",
            "task_id": task_id,
            "kind": None,
            "completed_ratio": completed_ratio,
            "message": message,
        }
        await create_queue_publish(queue_name, result)

    download_parallelism = max(1, int(os.getenv("BLOB_DOWNLOAD_MAX_CONCURRENCY", "8")))
    download_batch_size = max(download_parallelism, int(os.getenv("BLOB_DOWNLOAD_BATCH_SIZE", "16")))
    semaphore = asyncio.Semaphore(download_parallelism)

    async def download_one_file(name, index, total_files):
        local_target_path = os.path.join(tmp_source_pdf_folder, f"{name}")
        os.makedirs(os.path.dirname(local_target_path), exist_ok=True)
        blob_target_path = f"{temp}/{name}"
        try:
            async with semaphore:
                data = await asyncio.to_thread(blob_manager.get_data_from_blob, blob_target_path)
            if data:
                async with aiofiles.open(local_target_path, "wb") as download_file:
                    await download_file.write(data)
            else:
                logging.error(f"No data found for blob: {blob_target_path}")
            completed_ratio = min(10, 1 + int(index / max(1, total_files) * 9))
            await publish_download_update(
                f"Downloaded {index}/{total_files} files",
                completed_ratio,
            )
            return local_target_path
        except Exception as e:
            logging.error(f"Error downloading {blob_target_path}: {e}")
            return None

    try:
        total_files = len(revision_filenames)
        await publish_download_update(
            f"Downloading {total_files} input files",
            1,
        )
        local_paths = []
        # Run downloads in bounded batches to avoid blowing the HTTP connection pool.
        for batch_start in range(0, total_files, download_batch_size):
            batch_names = revision_filenames[batch_start: batch_start + download_batch_size]
            tasks = [
                download_one_file(name, batch_start + idx + 1, total_files)
                for idx, name in enumerate(batch_names)
            ]
            batch_paths = await asyncio.gather(*tasks)
            local_paths.extend(batch_paths)
        # Filter out failed downloads
        local_paths = [p for p in local_paths if p]
        if len(local_paths) != total_files:
            raise Exception("Some files failed to download.")
        #logging.debug("Files Fetched")
        await publish_download_update("All input files downloaded", 10)
    except Exception as e:
        logging.error(f"the error is {e}. The blob failed.")
        mongo_client.update_status_for_task_id(task_id, "FAILED_TO_DOWNLOAD_FILE", "FAILED")
        return False, False, False

    # Return the folder, the last file path, and output dir
    return tmp_source_pdf_folder, local_paths[-1], output_dir_path

import concurrent.futures

async def upload_file(task_id, created_time, json_path, html_path, type_filter):
    """
    Uploads processed files (JSON, HTML, Excel, images) to blob storage in parallel for maximum performance.
    Uses ThreadPoolExecutor for parallel file uploads, especially for large image folders.
    """
    Text, Images, Tables, Identical_Content = 0, 0, 0, 0
    mongo_client = MongoDB()
    try:
        temp = get_blob_segment(
            task_id,
            DELTA_COMPARATOR,
            created_time,
        )
        processed_dir = os.path.dirname(os.path.dirname(json_path))
        task_dir = os.path.dirname(processed_dir)
        blob_manager = BlobClientManager()

        blob_file_path = f"{temp}/change_tracer.json"

        # Process and upload Excel file
        try:
            status = await asyncio.to_thread(process_change_tracer, json_path)
            if status:
                excel_path = os.path.join(processed_dir, "change_tracer.xlsx")
                image_folder = processed_dir
                status = await asyncio.to_thread(
                    convert_change_tracer_json_to_excel,
                    json_path,
                    excel_path,
                    image_folder,
                )
                if status:
                    logging.info(f"Excel file created at {excel_path}")
                    excel_blob_path = f"{temp}/change_tracer.xlsx"
                    excel_data = await asyncio.to_thread(_read_binary_file, excel_path)
                    content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    await asyncio.to_thread(
                        blob_manager.upload_data_to_blob,
                        excel_blob_path,
                        excel_data,
                        content_type,
                        True,
                    )
        except Exception as e:
            logging.error(f"Error processing change tracer JSON {json_path}: {e}")

        if os.path.exists(json_path):
            Text, Images, Tables, Identical_Content = await asyncio.to_thread(count_extracted_data, json_path, task_id)
        # Upload JSON
        data = await asyncio.to_thread(_read_binary_file, json_path)
        content_type = "application/json"
        await asyncio.to_thread(
            blob_manager.upload_data_to_blob,
            blob_file_path,
            data,
            content_type,
            True,
        )

        # Upload HTML if present
        blob_file_path_html = f"{temp}/delta_comparator.html"
        if html_path and os.path.exists(html_path):
            data = await asyncio.to_thread(_read_binary_file, html_path)
            content_type = "text/html"
            await asyncio.to_thread(
                blob_manager.upload_data_to_blob,
                blob_file_path_html,
                data,
                content_type,
                True,
            )

        # Parallel image upload
        if "Image" in type_filter:
            image_folder = os.path.join(processed_dir, "images")
            #logging.debug(f"Image folder path : {image_folder}")
            if os.path.exists(image_folder):
                image_files = []
                for root, _, files in os.walk(image_folder):
                    for filename in files:
                        local_file_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(local_file_path, image_folder)
                        blob_file_path = f"{temp}/images/{relative_path.replace(os.sep, '/')}"
                        image_files.append((local_file_path, blob_file_path))

                def upload_one_image(args):
                    local_file_path, blob_file_path = args
                    try:
                        with open(local_file_path, "rb") as f:
                            content_type = "image/jpeg"  # Adjust as needed
                            blob_manager.upload_data_to_blob(blob_file_path, f.read(), content_type=content_type, processed=True)
                        #logging.debug(f"Uploaded image: {blob_file_path}")
                    except Exception as e:
                        logging.error(f"Failed to upload image {local_file_path}: {e}")

                # Use ThreadPoolExecutor for parallel uploads (I/O bound)
                max_workers = min(32, (os.cpu_count() or 2) * 4)
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    list(executor.map(upload_one_image, image_files, chunksize=16))
                #logging.debug(f"Uploaded image folder: {image_folder}")
            else:
                logging.warning(f"Image folder not found: {image_folder}")

        content = {
            "Images": Images,
            "Text": Text,
            "Tables": Tables,
            "Identical_Content": Identical_Content,
            "status": "SUCCESS"
        }
        mongo_client.update_content_for_task_id(task_id, content)

        completion_status = {
            "task_id": task_id,
            "status_code": 200,
            "status_message": "Result Files Upload Completed",
            "output_file_url_json": None,
            "Images:": Images,
            "Text:": Text,
            "Tables:": Tables,
            "Identical_Content:": Identical_Content,
            "output_file_url": "source_output.json",
            "collection_name": DELTA_COMPARATOR_COLLECTION,
        }
        print("Execution finished")
        tmp_task_folder = task_dir
        await asyncio.to_thread(shutil.rmtree, tmp_task_folder)

    except Exception as e:
        logging.error(f"Error in upload_file: {e}")
        mongo_client.update_status_for_task_id(task_id, "UPLOADING_TO_FAILED_FILES", "FAILED")
        completion_status = {
            "task_id": task_id,
            "status_code": 404,
            "status_message": "Failed to Upload Result file.",
            "output_file_url_json": None,
            "output_file_url": None,
            "collection_name": DELTA_COMPARATOR_COLLECTION,
        }
        pass

   

async def fix_xlsx_data(local_path, source_filename, revision_filenames, column_names): 
    column_names  = [name.strip() for name in column_names.split(",")] 
    df = pd.read_excel(os.path.join(local_path, source_filename))
    df.rename(columns={column_names[0]: "source"}, inplace=True)
    df.to_excel(os.path.join(local_path, source_filename), index=False)
    counter = 1
    for name in revision_filenames:
        df = pd.read_excel(os.path.join(local_path, name))
        df.rename(columns={column_names[counter]: "revised"}, inplace=True)
        df.to_excel(os.path.join(local_path, name), index=False)
        counter += 1


async def get_delta_comparator(task_details, task_id):
    mongo_client = MongoDB()
    queue_name = f"delta_comparator_{task_id}"
    async def _publish_eos_with_retry(queue_name: str, task_id: str, retries: int = 3):
        eos_payload = {"type": "EOS", "task_id": task_id}
        for attempt in range(1, retries + 1):
            try:
                await create_queue_publish(queue_name, eos_payload)
                return True
            except Exception as eos_error:
                logging.warning(
                    f"EOS publish failed for task_id={task_id} (attempt {attempt}/{retries}): {eos_error}"
                )
                if attempt < retries:
                    await asyncio.sleep(attempt * 0.5)
        return False

    task_id = task_id or task_details.get("task_id")
    queue_name = f"delta_comparator_{task_id}" if task_id else ""
    eos_published = False

    try:
        # Clear stale cache for this task id before processing.
        await _cleanup_task_cache(task_id)
        result = {
                    "type": "stream",
                    "task_id": task_id,
                    "kind": None,
                    "completed_ratio": 0,
                    "message": "Delta Comparator process started",
                }
        await create_queue_publish(queue_name, result)
        logging.info(f"get_delta_comparator started for task_id: {task_id}")
        task_id = task_details["task_id"]
        queue_name = f"delta_comparator_{task_id}"
        task_details = task_details["task_details"]
        revision_filenames = task_details["revision_filename"]
        created_time = task_details['created_timestamp']
        type = task_details.get("file_extension")
        type_filter = task_details['type_filter'] 
        

        mongo_client.update_status_for_task_id(task_id, "PROCESSING","STREAMING")
        local_source_path, local_conf_path, output_dir_path = await download_raw_files( task_id,  revision_filenames, created_time)
        app_folder = os.path.abspath(os.path.dirname(__file__))
        #logging.debug(f"Files downloaded at {local_source_path}")
        # Get the parent directory of app
        parent_folder = os.path.dirname(app_folder)
        tmp_source_pdf_folder = os.path.join(
            parent_folder, CACHE, DELTA_COMPARATOR , task_id, RAW
        )
        #logging.debug(f"tmp_source_pdf_folder {tmp_source_pdf_folder}")
        artifact_dir = os.path.join(
            parent_folder, CACHE, DELTA_COMPARATOR , task_id, PROCESSED,"artifacts"
        )
        #logging.debug(f"Artifact directory : {artifact_dir}")
        
        # return None
        local_revision_paths = []
        for name in revision_filenames:
            tmp_source_pdf_folder = os.path.join(
            parent_folder, CACHE, DELTA_COMPARATOR , task_id, RAW
        )
            local_revision_paths.append(os.path.join(tmp_source_pdf_folder, f"{name}"))
        key_column = task_details.get("key_column_mapping")
        if key_column and type.lower() == ".json":
            for path in local_revision_paths:
                    filename = os.path.basename(path)
                    key_column_name = key_column.get(filename)
                    if key_column_name and key_column_name != "text":
                        logging.info(f"transforming json file {filename} with key column {key_column_name}")
                        await transform_json(path, key_column_name)
        result = {
                    "type": "stream",
                    "task_id": task_id,
                    "kind": None,
                    "completed_ratio": 10,
                    "message": "Pipeline execution started",
                }
        queue_name = f"delta_comparator_{task_id}"
        await create_queue_publish(queue_name, result)
        logging.info("Pipeline execution started")
        try:
            await run_pipeline(local_revision_paths,type_filter,task_id,key_column)
        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}")
            completion_status = {
                        "task_id": task_id,
                        "status_code": 500,
                        "status_message": "Pipeline execution failed",
                        "output_file_url_json": None,
                        "output_file_url": None,
                        "collection_name": DELTA_COMPARATOR_COLLECTION,
                        "status": "FAILED"
            }
            await RabbitMQManager().publish_completion(task_id, completion_status)
            raise e
        logging.info("Pipeline execution finished")
        html_out = None
        input_file  = os.path.join(artifact_dir, "change_tracer.json")
        logging.info(f'Output file: {input_file}')
        result = {
                    "type": "stream",
                    "task_id": task_id,
                    "kind": None,
                    "completed_ratio": 95,
                    "message": "Pipeline execution finished and report generation started",
                }
        queue_name = f"delta_comparator_{task_id}"
        await create_queue_publish(queue_name, result)
        if False: # Skipping report generation for now as it is causing some issues and we want to unblock the flow. Will add it back once we have more stability in the pipeline.
            html_out = os.path.join(artifact_dir, "delta_comparator.html")
            try:
                generate_html_report(
                input_file,
                out_html_path=html_out,
                title="DELTA COMPARATOR: ",
                include_diff=False,
                color_changes=False,
                save_json_artifact=False,
                show_images=True,
                show_text=True,
                show_table_body=False                        
                )
            except Exception as e:
                logging.error(f"Error in generate_html_report: {e}")
                pass
        completion_status = {
            "task_id": task_id,
            "status_code": 200,
            "status_message": "Results Generation Completed",
            "output_file_url_json": None,
            "output_file_url": None,
            "collection_name": DELTA_COMPARATOR_COLLECTION,
        }
        await RabbitMQManager().publish_completion(task_id, completion_status)
        logging.info(f'Output file upload to File Storage started.')
        await upload_file(task_id, created_time,input_file,html_path=html_out, type_filter=type_filter)
        result = {
                    "type": "stream",
                    "task_id": task_id,
                    "kind": None,
                    "completed_ratio": 100,
                    "message": "Delta Comparator Process Completed",
                }
        queue_name = f"delta_comparator_{task_id}"
        await create_queue_publish(queue_name, result)
        completion_status = {
            "task_id": task_id,
            "status_code": 200,
            "status_message": "Delta Comparator Process Completed",
            "output_file_url_json": None,
            "output_file_url": None,
            "collection_name": DELTA_COMPARATOR_COLLECTION,
            "status": "SUCCESS"
        }
        mongo_client.update_status_for_task_id(task_id, "SUCCESS")
        #await RabbitMQManager().publish_completion(task_id, completion_status)
        logging.info(f"get_delta_comparator finished for task_id: {task_id}")
        eos_published = await asyncio.shield(_publish_eos_with_retry(queue_name, task_id))
    except Exception as error:
        logging.error(f"Error in get_delta_comparator: {error}")
        mongo_client.update_status_for_task_id(task_id, "FAILED")
        completion_status = {
                    "task_id": task_id,
                    "status_code": 500,
                    "status_message": "Delta Comparator Process Failed",
                    "output_file_url_json": None,
                    "output_file_url": None,
                    "collection_name": DELTA_COMPARATOR_COLLECTION,
                    "status": "FAILED"
        }
        await RabbitMQManager().publish_completion(task_id, completion_status)
    finally:
        # Always cleanup task-local cache to prevent unbounded disk growth.
        await _cleanup_task_cache(task_id)
        if task_id and queue_name and not eos_published:
            with suppress(Exception):
                await asyncio.shield(_publish_eos_with_retry(queue_name, task_id))

def count_extracted_data(json_path, task_id):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)["requirements"]
        text_count = 0
        image_count = 0
        table_count = 0
        # Count "Identical content" in change logs
        identical_count = 0
        for item in data:
            item_type = item.get("kind")
            if item_type == "text":
                text_count += 1
                change_log = item.get("Change_log", {})
                for version, version_data in change_log.items():
                    for pair in version_data.get("pairs", []):
                        if pair.get("change_type") == "No Change":
                            identical_count += 1
            elif item_type == "image":
                image_count += 1
            elif item_type == "table":
                table_count += 1
            
        logging.info(f"Extracted data counts - Text: {text_count}, Images: {image_count}, Tables: {table_count}, Identical: {identical_count} for task_id: {task_id}")
        return text_count, image_count, table_count, identical_count
    except Exception as e:
        logging.error(f"Error counting extracted data in {json_path} for task_id: {task_id}: {e}")
        return 0, 0, 0, 0

def process_change_tracer(json_file: str) -> None:
    try:
        # Load the JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        requirements = data.get('requirements', [])
        count = 0
        NEW_ORDER = 0

        for req in requirements:
            change_log = req.get('Change_log', {})
            count += 1
            versions = req.get("versions", {})
            for version_key, changes in change_log.items():
                pair = changes['pairs'][0]
                page_keys = list(pair["page_info"].keys())
                source_key, target_key = page_keys[0], page_keys[1]
                # Update change_log with required fields
                change_log[version_key] = pair
                pair["source_page"] = normalize_value(pair["page_info"].get(source_key) or pair["page_info"].get(target_key))
                pair["target_page"] = normalize_value(pair["page_info"].get(target_key) or pair["page_info"].get(source_key))
                pair["source_order"] = normalize_value(versions.get(source_key, {}).get("order") or versions.get(target_key, {}).get("order"))
                pair["target_order"] = normalize_value(versions.get(target_key, {}).get("order") or versions.get(source_key, {}).get("order"))
                try:
                    pair["pair_order"] = int(str(pair["pair_key"]).split("|")[0])
                except (ValueError, TypeError, KeyError):
                    print(pair)
                    pair["pair_order"] = ""
                del pair["page_info"]
                if pair["change_type"] == "New":
                    if NEW_ORDER == 0:
                        NEW_ORDER = count
                    if pair["target_order"] != "":
                        req["target_order"] = int(pair["target_order"])
                    else:
                        value = normalize_value(versions.get(target_key, {}).get("order"))
                        req["target_order"] = value or 1
                    req["present_order"] = int(pair["pair_order"])
            for version_key, version_data in versions.items():
                if "page" in version_data:
                    del version_data["page"]
                if "order" in version_data:
                    del version_data["order"]

                

        requirements_count = len(requirements)
        # Move requirements based on target_order
        for i in range(requirements_count - 1, NEW_ORDER - 2, -1):
            req = requirements[i]
            if "target_order" in req:
                target_idx = req["target_order"] + 1
                del req["target_order"]
                del req["present_order"]
                moved_req = requirements.pop(i)
                requirements.insert(target_idx, moved_req)

        data["requirements"] = requirements
        # Write the updated data back to the same file
        # Write the updated data back to a new file
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(traceback.format_exc())
        logging.error(f"Error processing change tracer JSON {json_file}: {e}")
        return False

def convert_change_tracer_json_to_excel(json_path, excel_path, images_folder):
    """
    Converts a change tracer JSON file to Excel, embedding images if present.

    Args:
        json_path (str): Path to the input JSON file.
        excel_path (str): Path to the output Excel file.
        images_folder (str): Folder where images are stored.
    """
    try:
        # Read the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)["requirements"]

        # Normalize (flatten) the nested dicts
        df = pd.json_normalize(data)

        # Save to Excel first (without images)
        df.to_excel(excel_path, index=False)

        # Find all columns that match the pattern 'versions.*.img_path'
        img_cols = [col for col in df.columns if re.match(r'versions\.\w+\.img_path', col)]
        if img_cols:
            wb = load_workbook(excel_path)
            ws = wb.active
            for img_col in img_cols:
                col_letter = chr(65 + df.columns.get_loc(img_col))
                for idx, img_path in enumerate(df[img_col], start=2):  # Excel rows start at 2 (header is 1)
                    if pd.notna(img_path) and isinstance(img_path, str) and img_path.strip():
                        full_img_path = os.path.join(images_folder, img_path) if not img_path.startswith(images_folder) else img_path
                        resolved_img_path = os.path.abspath(full_img_path)
                        img_basename = os.path.basename(resolved_img_path)
                        img_ext = os.path.splitext(img_basename)[1].lower()
                        exists = os.path.exists(resolved_img_path)
                        # logging.debug(
                        #     f"Excel image candidate row={idx} col={img_col} path={resolved_img_path} "
                        #     f"exists={exists} basename={img_basename} ext={img_ext} col_letter={col_letter}"
                        # )
                        if not exists:
                            logging.error(
                                f"Could not add image for row {idx}, column {img_col}: "
                                f"file not found at {resolved_img_path}"
                            )
                            continue
                        try:
                            img = XLImage(resolved_img_path)
                            ws.add_image(img, f"{col_letter}{idx}")
                            # Expand row height and column width to fit image
                            if img.height:
                                ws.row_dimensions[idx].height = img.height * 0.75
                            if img.width:
                                ws.column_dimensions[col_letter].width = img.width * 0.75
                        except Exception as e:
                            logging.error(f"Could not add image for row {idx}, column {img_col}: {e}")
            wb.save(excel_path)
            return True
    except Exception as e:
        logging.error(f"Error converting change tracer JSON to Excel: {e}")
        return False




def normalize_value(value):
    if value == "0.0":
        return 1
    if isinstance(value, str) and value.strip() == "":
        return ""
    
    try:
        return int(abs(float(value)))
    except (ValueError, TypeError):
        return ""