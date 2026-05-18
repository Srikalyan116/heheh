import requests
import asyncio
import aiohttp
import zipfile
import json
from pathlib import Path
from io import BytesIO
import os
from app.delta_comparator.utils.logger import log as logging
# API endpoint
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(BASE_DIR, "source.pdf")

submit_url = "https://llm-extractor-api-dev-ocp93330451.apps.nash.hco000141.mars.cloud.zf-world.com/api/pdf_extraction/task/"
status_url = "https://llm-extractor-api-dev-ocp93330451.apps.nash.hco000141.mars.cloud.zf-world.com/api/pdf_extraction/status/{task_id}/"
poll_interval = 5  # seconds


# def submit_pdf(pdf_path):
#     with open(pdf_path, "rb") as f:
#         files = {"file": (Path(pdf_path).name, f, "application/pdf")}
#         response = requests.post(submit_url, files=files)
#         response.raise_for_status()
#         return response.json()["task_id"]
def submit_pdf(path):
    headers = {}
    token=""
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # These are query parameters, not form data
    params = {
        "start_page": 10,
        "end_page": 10
    }

    data = {
        "username": "Anirban Lekharu",
        "email": "anirban.lekharu@zf.com"        
    }

    files = {
        "source_file": open(path, "rb")  # API expects this key
    }

    response = requests.post(submit_url, data=data, params=params, headers=headers, files=files)
    response.raise_for_status()
    return response.json()["task_id"]


async def poll_for_zip(task_id):
    async with aiohttp.ClientSession() as session:
        while True:
            url = status_url.format(task_id=task_id)
            async with session.get(url) as resp:
                if resp.status == 200 and resp.headers.get("Task-Status") == "Success":
                    return await resp.read(), task_id
                #logging.debug(f"[{task_id}] Still processing...")
            await asyncio.sleep(poll_interval)


def extract_first_json_from_zip(zip_bytes, out_json_path):
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
        for name in z.namelist():
            if name.endswith(".json"):
                with z.open(name) as f:
                    content = json.load(f)
                    with open(out_json_path, "w", encoding="utf-8") as out_f:
                        json.dump(content, out_f, indent=2, ensure_ascii=False)
                #logging.debug(f"Saved JSON to {out_json_path}")
                return
    raise Exception("No JSON found in ZIP")


async def main():
    pdf_path = source_path  # your PDF file
    task_id = submit_pdf(pdf_path)
    #logging.debug(f"Submitted task {task_id}")

    zip_data, task_id = await poll_for_zip(task_id)
    output_path = f"output_{task_id}.json"
    extract_first_json_from_zip(zip_data, output_path)


if __name__ == "__main__":
    asyncio.run(main())
