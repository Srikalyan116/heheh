import os
import json
import zipfile
import requests
import aiohttp
import asyncio
from io import BytesIO
from app.delta_comparator.utils.logger import log as logging
import loguru
import shutil
from urllib.parse import urljoin
from typing import Any

def _to_bool(v: Any, *, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int,)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    # Fallback: only True for truthy non-bool, non-str values
    return bool(v)

# safe list of image extensions we consider images
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".webp", ".bmp", ".svg"}

# logging already present in your file
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

class PDFExtractor:
    def __init__(self, submit_url, status_url, poll_interval=5, token="", username="", email="",image_text_extraction: bool = False,):
        self.submit_url = submit_url
        self.status_url = status_url
        self.poll_interval = poll_interval
        self.token = token
        self.username = username
        self.email = email
        #self.image_text_extraction =  bool(image_text_extraction)
        self.image_text_extraction = _to_bool(image_text_extraction, default=False)
    
    def _is_image_name(self, member_name: str) -> bool:
        """Return True if the zip member looks like an image filename."""
        ext = os.path.splitext(member_name)[1].lower()
        return ext in _IMAGE_EXTS
    
    def submit_pdf(self, path, start_page, end_page, timeout: int = 60):
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        # Query params (server expects these via Query)
        params = {}
        if start_page is not None:
            params["start_page"] = str(int(start_page))
        if end_page is not None:
            params["end_page"] = str(int(end_page))

        # Form fields (server expects image_text_extraction as Form)
        form_data = {
            "username": self.username,
            "email": self.email or "",
            "image_text_extraction": "true" if self.image_text_extraction else "false",
        }

        files = {
            "source_file": open(path, "rb"),           
        }
        #logging.debug(f"Submitting PDF {path} with pages {start_page}-{end_page}, image_text_extraction={self.image_text_extraction}")
        response = requests.post(self.submit_url, params=params, data=form_data, files=files, timeout=timeout)

        response.raise_for_status()
        j = response.json()
        return j.get("task_id") or j.get("TaskID") or j    

    async def poll_for_zip(self, task_id, label):
        async with aiohttp.ClientSession() as session:
            while True:
                url = self.status_url.format(task_id=task_id)
                async with session.get(url) as resp:
                    if resp.status == 200 and resp.headers.get("Task-Status") == "Success":
                        return await resp.read(), task_id
                    #logging.debug(f"[{task_id}] {label} PDF extracting ...")
                await asyncio.sleep(self.poll_interval)

    def extract_first_json(self, zip_bytes, out_path):
        with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
            for name in z.namelist():
                if name.endswith(".json"):
                    with z.open(name) as f:
                        content = json.load(f)
                        with open(out_path, "w", encoding="utf-8") as out_f:
                            json.dump(content, out_f, indent=2, ensure_ascii=False)
                    #logging.debug(f"Saved JSON to {out_path}")
                    return
        raise Exception("No JSON found in ZIP")
    
    def _extract_images_from_zipfileobj(self, zip_fileobj: zipfile.ZipFile, images_out_dir: str) -> list:
        """
        Extract image-like members from a ZipFile object; return list of filepaths extracted.
        This inspects all members (including ones within subfolders).
        """
        extracted = []
        for member in zip_fileobj.namelist():
            if member.endswith("/"):
                continue
            # basename may include directories: take basename for output filename
            basename = os.path.basename(member)
            if not basename:
                continue
            if self._is_image_name(basename):
                try:
                    target_path = os.path.join(images_out_dir, basename)
                    # avoid overwrite; if collision occurs, keep first extracted copy
                    if not os.path.exists(target_path):
                        with zip_fileobj.open(member) as rf, open(target_path, "wb") as wf:
                            shutil.copyfileobj(rf, wf)
                        #logging.debug(f"Extracted inner image {member} -> {target_path}")
                    else:
                        logging.debug(f"Image {basename} already exists; skipping overwrite.")
                    extracted.append(target_path)
                except Exception as e:
                    logging.warning(f"Failed to extract image member {member}: {e}")
        return extracted

    def extract_first_json_and_images(self, zip_bytes: bytes, out_dir: str, out_json_name: str, debug_save_zip: bool = False):
        """
        Robust extractor:
          - save debug ZIP if requested
          - extract JSON (first .json found anywhere in the zip) -> out_dir/out_json_name
          - extract images anywhere (files with image extensions) -> out_dir/images/<basename>
          - if nested zip members found (e.g. images.zip) we open them and extract images too
          - normalize img_path-like fields in JSON to "images/<basename>"
        Returns a tuple (out_json_path, list_of_extracted_image_basenames).
        """
        _ensure_dir(out_dir)
        images_out_dir = os.path.join(out_dir, "images")
        _ensure_dir(images_out_dir)

        # optional debug: keep raw ZIP
        if debug_save_zip:
            try:
                dbg_path = os.path.join(out_dir, "debug_raw.zip")
                with open(dbg_path, "wb") as dbgf:
                    dbgf.write(zip_bytes)
                #logging.debug(f"Wrote debug_raw.zip to {dbg_path}")
            except Exception as e:
                logging.warning(f"Could not write debug zip: {e}")

        with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
            members = z.namelist()
            #logging.debug(f"Top-level ZIP contains {len(members)} members; sample: {members[:20]}")

            # 1) Extract images present directly in top-level zip (in any folder)
            extracted_images = []
            for member in members:
                if member.endswith("/"):
                    continue
                normalized = member.replace("\\", "/")
                basename = os.path.basename(normalized)
                if not basename:
                    continue
                # skip nested zip files for now; handle them below
                if normalized.lower().endswith(".zip"):
                    continue
                if self._is_image_name(basename):
                    try:
                        target_path = os.path.join(images_out_dir, basename)
                        if not os.path.exists(target_path):
                            with z.open(member) as rf, open(target_path, "wb") as wf:
                                shutil.copyfileobj(rf, wf)
                            #logging.debug(f"Extracted top-level image {member} -> {target_path}")
                        else:
                            logging.debug(f"Top-level image {basename} already extracted.")
                        extracted_images.append(target_path)
                    except Exception as e:
                        logging.warning(f"Failed to extract top-level image {member}: {e}")

            # 2) If nested zip members are present (e.g., images.zip or a folder zipped), open and extract images from them
            for member in members:
                if member.endswith("/"):
                    continue
                normalized = member.replace("\\", "/")
                if normalized.lower().endswith(".zip"):
                    try:
                        nested_bytes = z.read(member)
                        with zipfile.ZipFile(BytesIO(nested_bytes), "r") as nested_z:
                            #logging.debug(f"Inspecting nested zip {member}; members: {nested_z.namelist()[:30]}")
                            nested_extracted = self._extract_images_from_zipfileobj(nested_z, images_out_dir)
                            extracted_images.extend(nested_extracted)
                    except Exception as e:
                        logging.warning(f"Failed to extract nested zip member {member}: {e}")

            # 3) If the zip contains a directory structure like "StellantisSource/..." with images inside,
            # the top-level image extraction loop already handled members with nested paths (basename extraction),
            # but we may also see images inside a nested folder. _extract_images_from_zipfileobj handles that.

            # 4) Find the first JSON member anywhere in the top-level zip (including subfolders)
            json_member = None
            for member in members:
                if member.lower().endswith(".json"):
                    json_member = member
                    break

            # If no JSON in top-level, check nested zips for JSON as well
            if not json_member:
                for member in members:
                    normalized = member.replace("\\", "/")
                    if normalized.lower().endswith(".zip"):
                        try:
                            nested_bytes = z.read(member)
                            with zipfile.ZipFile(BytesIO(nested_bytes), "r") as nested_z:
                                for nm in nested_z.namelist():
                                    if nm.lower().endswith(".json"):
                                        # use nested zip json; extract its bytes into content variable below
                                        json_member = (member, nm)  # indicate nested (store tuple)
                                        break
                                if json_member:
                                    break
                        except Exception:
                            continue

            if not json_member:
                raise Exception("No JSON found in ZIP")

            # 5) load JSON content (supports top-level json_member string or nested tuple)
            if isinstance(json_member, tuple):
                # nested zip scenario: json_member = (nested_zip_member_name, json_inside_name)
                nested_zip_member, inner_json_name = json_member
                nested_bytes = z.read(nested_zip_member)
                with zipfile.ZipFile(BytesIO(nested_bytes), "r") as nested_z:
                    with nested_z.open(inner_json_name) as jf:
                        content = json.load(jf)
            else:
                # typical top-level JSON file
                with z.open(json_member) as jf:
                    content = json.load(jf)

            # 6) Normalize img_path-like keys in JSON entries to "images/<basename>"
            def _fix_entry(e):
                if not isinstance(e, dict):
                    return
                for key in ("img_path", "image_path", "image", "img"):
                    val = e.get(key)
                    if isinstance(val, str) and val.strip():
                        orig = val.strip()
                        basename = os.path.basename(orig)
                        candidate_local = os.path.join(images_out_dir, basename)
                        # if we extracted this image, use relative path; otherwise still normalize to images/<basename>
                        if os.path.exists(candidate_local):
                            e[key] = os.path.join("images", basename).replace("\\", "/")
                        else:
                            e[key] = os.path.join("images", basename).replace("\\", "/")
                # recurse for nested dict/list structures
                for k, v in list(e.items()):
                    if isinstance(v, dict):
                        _fix_entry(v)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                _fix_entry(item)

            if isinstance(content, list):
                for entry in content:
                    _fix_entry(entry)
            elif isinstance(content, dict):
                _fix_entry(content)

            # 7) write normalized JSON to out_path
            out_path = os.path.join(out_dir, out_json_name)
            with open(out_path, "w", encoding="utf-8") as outf:
                json.dump(content, outf, indent=2, ensure_ascii=False)

            extracted_basenames = [os.path.basename(p) for p in extracted_images]
            #logging.debug(f"Saved JSON to {out_path}; extracted {len(extracted_basenames)} images into {images_out_dir}")
            return out_path, extracted_basenames

    # update extract_single_pdf to use above:
    async def extract_single_pdf(self, pdf_path, out_dir, page_range, label):
        """
        Submit PDF -> poll -> extract JSON + images.
        Returns: out_json_path (str)  — preserves backward compat for callers.
        """
        os.makedirs(out_dir, exist_ok=True)
        task_id = self.submit_pdf(pdf_path, *page_range)
        zip_bytes, _ = await self.poll_for_zip(task_id, label=label)
        out_json_name = f"{label}_{task_id}.json"
        # our extractor returns (out_path, extracted_basenames)
        result = self.extract_first_json_and_images(zip_bytes, out_dir, out_json_name, debug_save_zip=False)
        if isinstance(result, tuple):
            out_path, extracted_basenames = result
        else:
            out_path = result
        # Return only the json path (string)
        return out_path   
    
    async def extract_pdfs(self, source_pdf_path, revised_pdf_path, out_dir, source_page_range, revised_page_range):
        os.makedirs(out_dir, exist_ok=True)

        source_task_id = self.submit_pdf(source_pdf_path, *source_page_range)
        revised_task_id = self.submit_pdf(revised_pdf_path, *revised_page_range)

        source_zip, sid = await self.poll_for_zip(source_task_id, label="Source")
        revised_zip, rid = await self.poll_for_zip(revised_task_id, label="Revised")

        source_path = os.path.join(out_dir, f"source_{sid}.json")
        revised_path = os.path.join(out_dir, f"revised_{rid}.json")

        self.extract_first_json(source_zip, source_path)
        self.extract_first_json(revised_zip, revised_path)

        return source_path, revised_path
