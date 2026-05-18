import os
import json
import requests
from typing import Dict, Union, Optional
import asyncio
from app.delta_comparator.core.weaviate_wrapper import WeaviateClient

# import loguru
from app.delta_comparator.utils.logger import log as logging
import pandas as pd
from typing import List, Dict, Any
from openai import OpenAI, AsyncOpenAI
import re
import mlflow
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# loguru.logger.add("matching_logs.txt", format="{time} {level} {message}", level="DEBUG", rotation="1 MB")

class QueryProcessor:
    """Processes database queries from Weaviate and returns responses"""

    def __init__(self, collection_name: str):
        """
        :param query: The query string to be processed
        :param collection_name: The name of the target Weaviate collection
        """
        #self.query = query
        self.collection_name = collection_name
        self.client = WeaviateClient()
        # OpenAI-like client used for embeddings (same as in your other code)
        request_url = os.getenv("EMBED_SERVICE_URL") + "/v1"
        self.openai_client = OpenAI(
            base_url=request_url,
            api_key=os.getenv("EMBED_PWD"),
            timeout=1800,
        )
    async def get_batch_embeddings(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        embeddings: List[List[float]] = []
        total_tokens = 0
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            #logging.debug(f"Embedding batch {i // batch_size + 1} (size {len(batch)})")

            # run sync API call in a thread
            resp = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                model=os.getenv("EMBED_MODEL"),
                input=batch,
                encoding_format="float"
            )
            embeddings.extend([item.embedding for item in resp.data])
            
            if hasattr(resp, "usage"):
                    total_tokens += resp.usage.total_tokens
        logging.info(f"Total Query Tokens: {total_tokens} ")
        mlflow.set_tag("Models", os.getenv("EMBED_MODEL")) 
        return embeddings, total_tokens
    
    def extract_core_requirement(self, text: str) -> str:
        """Extract the core requirement part of the sentence."""
        parts = re.split(r"[.;]", text)
        for part in parts:
            if any(kw in part.lower() for kw in ["shall", "must", "should", "require"]):
                return part.strip()
        return text[:200].strip()  # fallback to first 200 chars    

    def is_false_positive(self, query: str, candidate: str, threshold: float = 0.3) -> bool:
        """
        Returns True if the candidate is likely a false positive based on Jaccard similarity.
        
        Args:
            query (str): The input query requirement.
            candidate (str): The matched candidate requirement.
            threshold (float): Similarity threshold (lower = more strict).
        
        Returns:
            bool: True if it's likely a false match.
        """
        query_words = set(re.findall(r"\b\w+\b", query.lower()))
        candidate_words = set(re.findall(r"\b\w+\b", candidate.lower()))

        if not query_words or not candidate_words:
            return True  # Avoid zero division or meaningless comparison

        intersection = query_words & candidate_words
        union = query_words | candidate_words

        jaccard_similarity = len(intersection) / len(union)

        return jaccard_similarity < threshold

    def normalize(self, text: str) -> str:
        """Normalize text for exact matching."""
        return " ".join(text.strip().split()).lower()

    async def get_response_async(self, query: str, embedding: List[float], similarity_threshold: float = 0.88) -> Optional[Dict[str, object]]:
        """
        Query weaviate with a prepared embedding and apply your matching heuristics.
        Returns dict {"source": ..., "score": ...} or None.
        """
        try:
            # Query Weaviate (assumes self.client.query_collection is async or awaitable)
            vectordb_response = await asyncio.to_thread(
               self.client.query_collection, 
                query_embed=embedding,
                query=query,
                target_collection_name=self.collection_name,
            )
                        
        except Exception as e:
            logging.error(f"Vector DB query error: {e}")
            return None

        if not vectordb_response:
            #logging.debug("✘ No results from vector DB")
            return None

        core_query = self.extract_core_requirement(query)
        if isinstance(core_query, dict):
            print(f"Warning: extract_core_requirement returned a dict for query '{core_query}'. This may cause issues in matching.")
        else:
            normalized_query = self.normalize(core_query)

        best_match = None
        best_score = 0.0

        for result in vectordb_response:
            source_text = result.get("properties", {}).get("source", "").strip()
            score = result.get("score", 0.0)

            core_source = self.extract_core_requirement(source_text)
            normalized_source = self.normalize(core_source)

            jaccard_fail = self.is_false_positive(core_query, core_source)

            #logging.debug(f"→ Candidate: {source_text[:80]}")
            #logging.debug(f"→ Score: {score:.4f} | Jaccard Fail: {jaccard_fail}")
            #logging.debug(f"→ Normalized Match? {normalized_query == normalized_source}")

            # exact normalized match -> immediate accept
            if normalized_query == normalized_source:
                return {"source": source_text, "score": score}

            # otherwise check threshold and heuristics
            if score >= similarity_threshold and not jaccard_fail:
                if score > best_score:
                    best_match = {"source": source_text, "score": score}
                    best_score = score

        if best_match:
            return best_match

        return None
    
    async def batch_query(self, queries: List[str], embeddings: List[List[float]], max_concurrency: int = 16) -> List[Optional[Dict[str, object]]]:
        """
        Run many get_response_async calls concurrently, limited by a semaphore.
        Returns results in the same order as queries/embeddings.
        """
        if len(queries) != len(embeddings):
            raise ValueError("queries and embeddings must be same length")

        sem = asyncio.Semaphore(max_concurrency)

        async def _limited(i: int):
            async with sem:
                try:
                    return await self.get_response_async(queries[i], embeddings[i])
                except Exception as e:
                    logging.error(f"Error on query index {i}: {e}")
                    return None

        tasks = [asyncio.create_task(_limited(i)) for i in range(len(queries))]
        results = await asyncio.gather(*tasks)
        return results
