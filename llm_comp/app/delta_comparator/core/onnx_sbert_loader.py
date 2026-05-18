###Step2: Loader
import os
from app.delta_comparator.core.onnx_sbert import ONNXSentenceTransformer
from app.delta_comparator.utils.logger import log as logging
#from onnx_sbert import ONNXSentenceTransformer


_model = None

logging.debug(f"ONNX raw {os.environ.get('USE_ONNX')}")
logging.debug(f"ONNX evaluated {os.environ.get('USE_ONNX', '1') == '1'}")

def _is_onnx_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "model.onnx"))

def get_sbert_model():
    global _model

    if _model is None:
        model_path = os.environ.get(
            "SBERT_ONNX_DIR",
            "models/onnx-model"
        )
        logging.info(f"[ONNX] Loading model from: {model_path}")
        _model = ONNXSentenceTransformer(model_path)
    return _model


_get_sbert_model = get_sbert_model


def _trim_sbert_model_cache():
    global _model

    if _model is None:
        return None

    _model.cleanup_cache()
    return _model


def _clear_sbert_model_cache():
    global _model

    if _model is None:
        return None

    _model.clear_cache()
    return _model
