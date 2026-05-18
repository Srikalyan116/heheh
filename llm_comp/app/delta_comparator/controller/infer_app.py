"""To compare documents and find the revision changes using this api."""
from fastapi import APIRouter, HTTPException, Request
import asyncio
import os
import logging
from dotenv import load_dotenv, find_dotenv

try:
    from core.revision_comparer import RevisionComparer
    from core.utility import change_tracer_helper
except:
    from app.requirement_comparision_tracer.core.revision_comparer import RevisionComparer
    from app.requirement_comparision_tracer.core.utility import change_tracer_helper

# Load environment variables from the .env file
load_dotenv(dotenv_path="tracer.env")
load_dotenv(find_dotenv(".env", raise_error_if_not_found=True))
# Access the environment variables
model_name = os.getenv("MODEL_NAME")

router = APIRouter()

# Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

comparer = RevisionComparer()


@router.post("/compare_revisions")
async def compare_revisions(request: Request):
    """Compare revisions using the RevisionComparer class.

    Args:
        request (RevisionInput): JSON payload with source and revision file paths.

    Raises:
        HTTPException: If the input data is invalid or an error occurs during processing.

    Returns:
        dict: Message and output file path.
    """
    input_ = await request.json()
    try:
        result = await asyncio.to_thread(change_tracer_helper, input_)
        return result

    except Exception as e:
        #logger.error(f"Exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {str(e)}")
