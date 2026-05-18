# # app.py
# import os
# import shutil
# import asyncio
# import uuid
# import json
# from typing import List, Optional, Union
# from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from fastapi import FastAPI, HTTPException
# import os, json
# import loguru

# try:    
#     from core.run_pipeline import run_pipeline
#     from core.llm_judge import evaluate_artifact_with_llm
# except:
#     from app.delta_comparator.core.run_pipeline import run_pipeline
#     from app.delta_comparator.core.run_pipeline import evaluate_artifact_with_llm


# app = FastAPI(title="Delta Comparator API", version="1.0")
# @app.get("/")
# async def root():
#     return {"message": "Delta Comparator API is running. Visit /docs for Swagger UI."}

# TASKS = {}
# BASE_DIR = "uploads"
# os.makedirs(BASE_DIR, exist_ok=True)

# # Load rules.json once
# RULES_PATH = os.path.join("core", "rules.json")
# with open(RULES_PATH, "r") as f:
#     RULES = json.load(f)


# class TaskStatus(BaseModel):
#     task_id: str
#     status: str
#     result: dict | None = None

# def _to_bool(v: Optional[Union[bool, str, int]], default: bool = False) -> bool:
#     """Robust conversion for many common inputs."""
#     if v is None:
#         return default
#     if isinstance(v, bool):
#         return v
#     if isinstance(v, (int, float)):
#         return bool(v)
#     s = str(v).strip().lower()
#     if s in {"1", "true", "t", "yes", "y", "on"}:
#         return True
#     if s in {"0", "false", "f", "no", "n", "off", ""}:
#         return False
#     return default


# @app.post("/tasks", response_model=TaskStatus)
# async def create_task(
#     background_tasks: BackgroundTasks,
#     source_file: UploadFile = File(..., description="Upload the source file (.xlsx, .reqifz, .json, .pdf)"),
#     revised_files: List[UploadFile] = File(..., description="Upload one or more revised files"),
#     source_column: Optional[str] = Form(None, description="Column name to use from the source Excel file (if .xlsx)"),
#     revised_columns: Optional[str] = Form(None, description="Comma-separated column names for revised Excel files (e.g. 'ColumnA,ColumnB')"),
#     source_start_page: Optional[int] = Form(None, description="Start page for the source PDF (if .pdf)"),
#     source_end_page: Optional[int] = Form(None, description="End page for the source PDF (if .pdf)"),
#     revised_start_pages: Optional[str] = Form(None, description="Comma-separated start pages for revised PDFs (e.g. '4,10,20' or '4')"),
#     revised_end_pages: Optional[str] = Form(None, description="Comma-separated end pages for revised PDFs (e.g. '8,15,25' or '8')"),
#     username: Optional[str] = Form(None, description="Username for PDF extraction"),
#     email: Optional[str] = Form(None, description="Email for PDF extraction"),
#     #image_text_extraction: Optional[bool] = Form(None, description="If true, enable OCR/image text extraction for PDFs (only used for PDF inputs)"),
#     image_text_extraction: Optional[Union[bool, str]] = Form(None, description="Enable OCR? true/false"),
# ):
#     """
#     Create a processing task. `image_text_extraction` is optional and only applied when both
#     source and revised files are PDFs. For non-PDF inputs it is ignored.
#     """
#     task_id = str(uuid.uuid4())
#     task_dir = os.path.join(BASE_DIR, task_id)
#     os.makedirs(task_dir, exist_ok=True)
    
#     # Save source file
#     source_path = os.path.join(task_dir, source_file.filename)
#     with open(source_path, "wb") as f:
#         shutil.copyfileobj(source_file.file, f)

#     # Save revised files
#     revised_paths = []
#     for f in revised_files:
#         path = os.path.join(task_dir, f.filename)
#         with open(path, "wb") as out:
#             shutil.copyfileobj(f.file, out)
#         revised_paths.append(path)

#     # Parse revised_columns into list (trim whitespace)
#     revised_columns_list = [c.strip() for c in (revised_columns or "").split(",") if c.strip()] if revised_columns else []

#     # Determine whether this is a PDF workflow
#     is_pdf_workflow = source_path.lower().endswith(".pdf") and all(p.lower().endswith(".pdf") for p in revised_paths)

#     # Default values
#     source_page_range = None
#     revised_page_ranges: List[tuple] = []
    
#     raw_image_flag = image_text_extraction
#     normalized_flag = _to_bool(raw_image_flag, default=False)

#     # Ensure the name exists on every code path so closures won't break
#     use_image_text_extraction: bool = False

#     # PDF-specific parsing/validation
#     if is_pdf_workflow:
#         # validate source page pair (both or none)
#         if (source_start_page is None) ^ (source_end_page is None):
#             raise HTTPException(status_code=400, detail="Both source_start_page and source_end_page must be provided together for PDFs.")
#         if source_start_page is not None and source_end_page is not None:
#             if source_start_page <= 0 or source_end_page <= 0 or source_end_page < source_start_page:
#                 raise HTTPException(status_code=400, detail="Invalid source page range.")
#             source_page_range = (int(source_start_page), int(source_end_page))

#         # helper to parse comma-separated ints
#         def parse_int_list(s: Optional[str]) -> List[int]:
#             if not s or not s.strip():
#                 return []
#             parts = [p.strip() for p in s.split(",") if p.strip() != ""]
#             try:
#                 return [int(x) for x in parts]
#             except ValueError:
#                 raise HTTPException(status_code=400, detail="Revised page ranges must be integers or comma-separated integers.")

#         start_list = parse_int_list(revised_start_pages)
#         end_list = parse_int_list(revised_end_pages)

#         # if single pair provided and there are multiple revised files, expand to all revised files
#         if start_list and end_list and len(start_list) == 1 and len(end_list) == 1 and len(revised_paths) > 1:
#             start_list = [start_list[0]] * len(revised_paths)
#             end_list = [end_list[0]] * len(revised_paths)

#         if (bool(start_list) ^ bool(end_list)):
#             raise HTTPException(status_code=400, detail="Both revised_start_pages and revised_end_pages must be provided together for PDFs.")
#         if start_list and len(start_list) != len(end_list):
#             raise HTTPException(status_code=400, detail="Number of revised start pages and end pages must match.")
#         if start_list and len(start_list) != len(revised_paths):
#             raise HTTPException(status_code=400, detail="Number of revised page ranges must match number of revised files.")

#         if start_list:
#             # validate each pair
#             for s, e in zip(start_list, end_list):
#                 if s <= 0 or e <= 0 or e < s:
#                     raise HTTPException(status_code=400, detail="Invalid revised page range values.")
#             revised_page_ranges = list(zip(start_list, end_list))

#         # image_text_extraction only meaningful for PDFs; default to False if not provided
#         # use_image_text_extraction = bool(image_text_extraction) if image_text_extraction is not None else False
#         # print(f"Image flag 2: {image_text_extraction}")
#         use_image_text_extraction = normalized_flag
#         loguru.logger.debug("image_text_extraction: raw=%r normalized=%s", raw_image_flag, use_image_text_extraction)
#     else:
#         # Not a PDF workflow — ignore image_text_extraction but warn
#         if raw_image_flag is not None:
#             loguru.logger.warning("image_text_extraction provided but inputs are not PDFs; value will be ignored.")

#     # Init task status
#     TASKS[task_id] = {"status": "pending", "result": None}

#     async def run_task():
#         try:
#             result = await run_pipeline(
#                 source_path,
#                 revised_paths,
#                 task_id,
#                 RULES,
#                 source_column=source_column,
#                 revised_columns=revised_columns_list,
#                 source_page_range=source_page_range,
#                 revised_page_ranges=revised_page_ranges,
#                 username=username,
#                 email=email,
#                 image_text_extraction=use_image_text_extraction,
#             )
#             TASKS[task_id] = {"status": "completed", "result": result}
#         except Exception as e:
#             TASKS[task_id] = {"status": "failed", "result": {"error": str(e)}}

#     # Launch pipeline in background using the running event loop
#     # (this is fine as long as your pipeline is async-compatible).
#     asyncio.create_task(run_task())

#     return TaskStatus(task_id=task_id, status="pending")

# @app.get("/tasks/{task_id}", response_model=TaskStatus)
# async def get_task_status(task_id: str):
#     task = TASKS.get(task_id)
#     if not task:
#         return JSONResponse(status_code=404, content={"error": "Task not found"})
#     return TaskStatus(task_id=task_id, status=task["status"], result=task["result"])

# @app.post("/eval/{task_id}")
# async def evaluate_task(task_id: str):
#     """
#     Run LLM-as-judge evaluation on a finished pipeline task.
#     """
#     task = TASKS.get(task_id)
#     if not task:
#         raise HTTPException(status_code=404, detail="Task not found")

#     if task["status"] != "completed":
#         raise HTTPException(status_code=400, detail="Task is not completed yet")

#     artifact = task["result"]
#     if not artifact:
#         raise HTTPException(status_code=500, detail="No artifact available for evaluation")

#     # Run evaluation
#     eval_result = await evaluate_artifact_with_llm(artifact, RULES, task_id)

#     return eval_result
