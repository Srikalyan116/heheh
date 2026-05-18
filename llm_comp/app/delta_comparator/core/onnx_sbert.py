import numpy as np
import time
import os
from collections import OrderedDict
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from app.delta_comparator.utils.logger import log as logging


class ONNXSentenceTransformer:
    def __init__(self, model_path: str):
        self.model_path = model_path

        # -------- TOKENIZER --------
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True
        )

        # -------- PROVIDER SELECTION --------
        preferred_provider = "CPUExecutionProvider"
        configured_provider = os.environ.get("SBERT_ONNX_PROVIDER")
        if configured_provider and configured_provider != preferred_provider:
            logging.info(
                f"[ONNX] Ignoring SBERT_ONNX_PROVIDER={configured_provider}; forcing CPUExecutionProvider"
            )

        # -------- MODEL LOAD --------
        try:
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                model_path,
                provider=preferred_provider,
            )
        except Exception as e:
            logging.warning(f"Falling back to CPUExecutionProvider due to: {e}")
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                model_path,
                provider=preferred_provider,
            )

        logging.info(f"[ONNX] Model loaded with provider: {preferred_provider}")

        # -------- SETTINGS --------
        self.default_batch_size = 64 # int(os.getenv("ONNX_BATCH_SIZE", "64"))
        self.max_cache_size = int(os.getenv("ONNX_EMBED_CACHE_SIZE", "50000"))
        self.max_cache_bytes = int(
            os.getenv("ONNX_EMBED_CACHE_BYTES", str(256 * 1024 * 1024))
        )
        self.use_fp16 = os.getenv("ONNX_USE_FP16", "false").lower() == "true"

        # -------- CACHE (LRU) --------
        self._embedding_cache = OrderedDict()
        self._embedding_cache_bytes = 0

        # -------- CLEANUP CONTROL --------
        self._last_cleanup = time.time()
        self.cleanup_interval = int(os.getenv("ONNX_CACHE_CLEANUP_INTERVAL", "300"))

        # -------- SAFETY --------
        self.max_key_length = int(os.getenv("ONNX_MAX_KEY_LENGTH", "1000"))

    # -------- MEAN POOLING --------
    @staticmethod
    def _mean_pooling(token_embeddings, attention_mask):
        mask = attention_mask.astype(np.float32)
        mask_sum = np.sum(mask, axis=1, keepdims=True)
        mask_sum[mask_sum == 0] = 1e-9
        return np.sum(token_embeddings * mask[:, :, None], axis=1) / mask_sum

    @staticmethod
    def _estimate_cache_entry_bytes(text, embedding):
        text_bytes = len((text or "").encode("utf-8", errors="ignore"))
        emb_bytes = int(getattr(embedding, "nbytes", 0))
        return text_bytes + emb_bytes + 128

    def _pop_oldest_cache_entry(self):
        text, embedding = self._embedding_cache.popitem(last=False)
        self._embedding_cache_bytes -= self._estimate_cache_entry_bytes(text, embedding)
        if self._embedding_cache_bytes < 0:
            self._embedding_cache_bytes = 0

    def _store_cache_entry(self, text, embedding):
        existing = self._embedding_cache.pop(text, None)
        if existing is not None:
            self._embedding_cache_bytes -= self._estimate_cache_entry_bytes(text, existing)
        self._embedding_cache[text] = embedding
        self._embedding_cache_bytes += self._estimate_cache_entry_bytes(text, embedding)

    def _enforce_cache_limits(self, target_size=None, target_bytes=None):
        target_size = self.max_cache_size if target_size is None else target_size
        target_bytes = self.max_cache_bytes if target_bytes is None else target_bytes
        while self._embedding_cache and (
            len(self._embedding_cache) > target_size or self._embedding_cache_bytes > target_bytes
        ):
            self._pop_oldest_cache_entry()

    # -------- CACHE CLEANUP --------
    def _cleanup_cache(self):
        target_size = int(self.max_cache_size * 0.8)
        target_bytes = int(self.max_cache_bytes * 0.8)
        self._enforce_cache_limits(target_size=target_size, target_bytes=target_bytes)

        logging.debug(
            f"[ONNX] Cache cleanup done. entries={len(self._embedding_cache)} bytes={self._embedding_cache_bytes}"
        )

    def cleanup_cache(self):
        self._cleanup_cache()
        self._last_cleanup = time.time()

    # -------- MANUAL CLEAR --------
    def clear_cache(self):
        self._embedding_cache.clear()
        self._embedding_cache_bytes = 0
        self._last_cleanup = time.time()
        logging.info("[ONNX] Embedding cache cleared")

    # -------- MAIN ENCODE --------
    def encode(
        self,
        sentences,
        batch_size: int = None,
        normalize: bool = True,
        show_progress: bool = False,
        show_progress_bar=None,
        **kwargs
    ):
        if show_progress_bar is not None:
            show_progress = bool(show_progress_bar)

        normalize = kwargs.pop("normalize_embeddings", normalize)
        kwargs.pop("convert_to_numpy", None)
        kwargs.pop("show_progress_bar", None)
        kwargs.pop("show_progress", None)
        if kwargs:
            logging.debug(f"[ONNX] Ignoring unsupported encode kwargs: {sorted(kwargs.keys())}")

        if isinstance(sentences, str):
            sentences = [sentences]

        # Normalize inputs
        sentences = [
            "" if s is None else str(s)[:self.max_key_length]
            for s in sentences
        ]

        if not sentences:
            return np.empty((0, 768), dtype=np.float32)

        batch_size = batch_size or self.default_batch_size
        total_start = time.time()

        # -------- PERIODIC CLEANUP --------
        if time.time() - self._last_cleanup > self.cleanup_interval:
            self._cleanup_cache()
            self._last_cleanup = time.time()

        # -------- DEDUP --------
        unique_sentences = list(dict.fromkeys(sentences))
        for text in unique_sentences:
            if text in self._embedding_cache:
                self._embedding_cache.move_to_end(text)
        missing = [s for s in unique_sentences if s not in self._embedding_cache]

        # -------- ENCODE MISSING --------
        if missing:
            for i in range(0, len(missing), batch_size):
                batch = missing[i:i + batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="np"
                )

                outputs = self.model(**inputs)

                embeddings = self._mean_pooling(
                    outputs.last_hidden_state,
                    inputs["attention_mask"]
                )

                if normalize:
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    np.maximum(norms, 1e-12, out=norms)
                    embeddings /= norms

                # Reduce memory if enabled
                dtype = np.float16 if self.use_fp16 else np.float32
                embeddings = embeddings.astype(dtype, copy=False)

                # Store in LRU cache
                for text, emb in zip(batch, embeddings):
                    self._store_cache_entry(text, emb)

                # Enforce cache size
                self._enforce_cache_limits()

        # -------- BUILD OUTPUT (NO VSTACK) --------
        dim = next(iter(self._embedding_cache.values())).shape[0]
        dtype = np.float16 if self.use_fp16 else np.float32

        result = np.empty((len(sentences), dim), dtype=dtype)

        for i, s in enumerate(sentences):
            result[i] = self._embedding_cache[s]

        total_time = time.time() - total_start
        logging.debug(f"[ONNX] Encoded {len(sentences)} sentences in {total_time:.3f}s")

        return result