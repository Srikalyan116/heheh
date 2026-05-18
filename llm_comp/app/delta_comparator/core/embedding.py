"""Provides class to ingest content of csv to vector database."""

import json
import os
from typing import Dict, List
import asyncio
import pandas as pd
import requests
import time
from openai import AsyncOpenAI, OpenAI
from app.delta_comparator.utils.logger import log as logging
import mlflow
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from app.delta_comparator.core.weaviate_wrapper import WeaviateClient


class Vectorize:
    """
    Process CSV files and ingest the data into a Weaviate vector database.

    :param input_csv_file_path: The path to the input CSV file.
    :param collection_name: The name of the Weaviate collection.
    :param client (WeaviateClient): The client for vector database.
    """

    def __init__(self, file_path: str, collection_name: str):
        """
        Initialize the CSV2Vectordb with the CSV file path and collection name.

        :param file_path: path to the summary csv file.
        :param collection_name: collection name.
        """
        self.file_path = file_path
        self.collection_name = collection_name        
        self.client = WeaviateClient()
        self._setup_weaviate_collection()
        
        self.request_url = os.getenv("EMBED_SERVICE_URL") + "/v1"
        self.openai_client = OpenAI(
                        base_url=self.request_url,
                        api_key=os.getenv("EMBED_PWD"),
                        timeout=1800,
                    )

    def _setup_weaviate_collection(self) -> None:
        """
        Set up the Weaviate collection. If the collection already exists,
          it exits.
        """
        if self.client.collection_exists(self.collection_name):
            ##logging.debug(f"{self.collection_name} already exists!! ..")
            self.client.delete_collection(self.collection_name)
            #logging.debug(f"Recreated collection: {self.collection_name}")
            self.client.create_collection(self.collection_name)
        else:
            #logging.debug(f"{self.collection_name} does not exist! Creating ..")
            self.client.create_collection(self.collection_name)
       
    def _format_data(
        self,
        sources: List[str],
        embeddings: List[List[float]],
    ) -> Dict:
        """
        Format the data for insertion into Weaviate.

        :param sections: List of section.
        :param contents: List of text contents.
        :param embeddings: List of code chunk summary embeddings.

        :return: Dictionary containing formatted data.
        """
        data_obj = [
            {
                "source": source,
                "embedding": embedding,
            }
            for source, embedding in zip(
                sources, embeddings
            )
        ]
        return {"collection_name": self.collection_name, "data": data_obj}
    
    async def create_embedding(self,source):
        response = self.openai_client.embeddings.create(
            model=os.getenv("EMBED_MODEL"),
            input=source,
            encoding_format="float"
        )       
        
        return response    
     
    async def process_single_section(self, sources):
        """
        Process a single section and insert the data into Weaviate.

        :param section: The section.
        :param content: The content of the section.
        """
        total_tokens = 0
        try:
            for i in range(0,len(sources),10):                
                batch = sources[i:i+10]
                #logging.debug(f"Processing batch {i + 1}")
                # print(batch)

                start_time = time.time()
                embedding_response = await asyncio.create_task(self.create_embedding(batch))
                # print(f"Embeddings: {embedding_response}")
                embeddings = [item.embedding for item in embedding_response.data]                
                embed_duration = time.time() - start_time
                #logging.debug(f"Embedding time for batch {i + 1}: {embed_duration:.2f}s")

                if hasattr(embedding_response, "usage"):
                    total_tokens += embedding_response.usage.total_tokens

            # ----------- Format and Accumulate Data -----------
                for j in range(len(batch)):
                    source = batch[j]
                    embedding = embeddings[j]
                    formatted = self._format_data([source], [embedding])

                    start_time = time.time()
                    self.client.insert_data(formatted)   # Insert immediately
                    db_duration = time.time() - start_time
                    #logging.debug(f"Pushed item {j+1} of batch {i // 10 + 1} to DB in {db_duration:.2f}s")
            
        except ValueError as e:
            logging.warning(e)
        #logging.debug(f"Tokens Embedding: {total_tokens}")
        return total_tokens

    async def process_all_sections(self) -> None:
        sources = []
        if self.file_path.endswith(".csv"):
            df = pd.read_csv(self.file_path)
            sources = [row["source"] for _, row in df.iterrows()]

        elif self.file_path.endswith(".json"):
            import json
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):  # list of dicts
                sources = [row["source"] for row in data if "source" in row]
            elif isinstance(data, dict) and "source" in data:  # single dict
                sources = [data["source"]]

        else:
            raise ValueError("Unsupported file format. Only CSV and JSON are supported.")

        total_tokens = 0
        total_tokens = await self.process_single_section(sources)

        # log totals to MLflow
        mlflow.log_metric("ingestion_total_tokens", total_tokens)
        mlflow.set_tag("Models", os.getenv("EMBED_MODEL"))
        return total_tokens
