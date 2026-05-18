"""Module for interacting with Weaviate vector database.
This module provides functions and classes for interacting with Weaviate, 
including creating a client, adding data, querying, and more.
"""

import os
import uuid
from typing import Dict, List, Any

import weaviate
import weaviate.classes as wvc

# from dotenv import load_dotenv
# from langfuse.decorators import langfuse_context, observe
from weaviate.util import generate_uuid5
from app.delta_comparator.utils.logger import log as logging

from dotenv import load_dotenv, find_dotenv
env_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir,".env")
load_dotenv(env_path)


class WeaviateClient:
    """
    Class responsible for interacting with a Weaviate database.
    """

    def __init__(self):
        """
        Initialize the Weaviate client connection.
        :param None
        :return: None
        """
        self.client = weaviate.WeaviateClient(
            connection_params=weaviate.connect.ConnectionParams.from_params(                
                # http_host=os.getenv("RC_WEAVIATE_CLUSTER_HOST"),
                # http_port=int(os.getenv("RC_WEAVIATE_HTTP_PORT")),
                # http_secure=False,
                # grpc_host=os.getenv("RC_WEAVIATE_GRPC_HOST"),  # GRPC_HOST,
                # grpc_port=int(os.getenv("RC_WEAVIATE_GRPC_PORT")),
                # grpc_secure=False,
                http_host=os.getenv("WEAVIATE_HTTP_HOST"),
                http_port=os.getenv("WEAVIATE_HTTP_PORT"),
                http_secure=os.getenv("WEAVIATE_HTTP_SECURE"),
                grpc_host=os.getenv("WEAVIATE_GRPC_HOST"),
                grpc_port=os.getenv("WEAVIATE_GRPC_PORT"),
                grpc_secure=os.getenv("WEAVIATE_GRPC_SECURE"),
            ),
            additional_config=weaviate.config.AdditionalConfig(
                timeout=(1800, 1800),
            ),
            skip_init_checks=True,
        )
        self.client.connect()
        self.collection = None

    def __del__(self):
        self.client.close()

    def delete_collection(self, name: str) -> None:
        """
        Delete a collection in Weaviate DB.

        :param name: The name of the collection to delete.
        :return: None
        """
        self.client.collections.delete(name)
        #logging.debug(f"{name} collection deleted!")

    def list_collections(self) -> None:
        """
        List all collections in Weaviate DB.

        :param None
        :return: None
        """
        #logging.debug(self.client.collections.list_all())

    def collection_exists(self, name: str) -> bool:
        """
        Check if a collection exists in Weaviate DB.

        :param name: The name of the collection to check.
        :return: True if the collection exists, False otherwise.
        """
        return self.client.collections.exists(name)

    def create_collection(self, name: str) -> None:
        """
        Create a collection in Weaviate DB.

        :param name: The name of the collection to create.
        :return: None
        """
        self.collection = self.client.collections.create(
            name=name,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            properties=[
                wvc.config.Property(
                    name="source",
                    data_type=wvc.config.DataType.TEXT,
                    vectorize_property_name=False,
                ),
                # wvc.config.Property(
                #     name="section",
                #     data_type=wvc.config.DataType.TEXT,
                #     vectorize_property_name=False,
                # ),
                # wvc.config.Property(
                #     name="subsection",
                #     data_type=wvc.config.DataType.TEXT,
                #     vectorize_property_name=False,
                # ),
                # wvc.config.Property(
                #     name="subsubsection",
                #     data_type=wvc.config.DataType.TEXT,
                #     vectorize_property_name=False,
                # ),
            ],
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE
            ),
            inverted_index_config=wvc.config.Configure.inverted_index(
                index_null_state=True,
                index_property_length=True,
                index_timestamps=True,
            ),
        )

        #logging.debug(name, "collection created!")

    def get_collection(self, name: str) -> bool:
        """
        Get a collection from Weaviate DB.

        :param name: The name of the collection to retrieve.
        :return: The collection object if it exists, False otherwise.
        """
        if self.collection_exists(name):
            return self.client.collections.get(name)
        else:
            return False

    def insert_data(self, data: Dict) -> None:
        """
        Insert data into a Weaviate collection.

        :param name: The name of the collection to insert into.
        :param data: A list of dictionaries containing the data to insert.
        :return: None
        """
        name = data["collection_name"]
        data_obj = data["data"]
        collection = self.get_collection(name)
        data_objects = []
        for curr_val in data_obj:
            data_object = wvc.data.DataObject(
                properties={
                    "source": curr_val["source"],  #
                    # "section": curr_val["section"],
                    # "subsection": curr_val["subsection"],
                    # "subsubsection": curr_val["subsubsection"],
                },
                vector=curr_val["embedding"],
                uuid=generate_uuid5(curr_val["source"]),
            )
            data_objects.append(data_object)
            try:
                collection.data.insert_many(data_objects)
                return True
            except weaviate.exceptions.WeaviateInsertManyAllFailedError as e:
                logging.error(
                    f"Failed to insert data into collection {name}. Error: {e}"
                )
                return False

    def query_collection(
        self, query_embed: List[float], query: str, target_collection_name: str
    ) -> str:
        """
        Query the specified collection in Weaviate.

        Args:
            query_embed (List[float]): The embedding of the query.
            query (str): The query string.
            target_collection_name (str): The name of the target collection.

        Returns:
            bool: True if the query was successful, False otherwise.
        """
        collection = self.get_collection(target_collection_name)
        query_template = {
            "collection_name": target_collection_name,
            "query": query,
            "query_vector": query_embed,
            "query_properties": ["text"],
            #"alpha": 0.15, #Worked for General Motorss
            #"alpha": 0.85,  # More syntactic focus
            #"alpha": 0.40
            "alpha": 0.15,  # More syntactic focus
            # • alpha = 0.0 → Pure lexical (keyword) match
            # • alpha = 1.0 → Pure semantic (meaning) match
            # • In between → Combine both for better results
            #"top_n": 7,
            "top_n": 3,
            
        }
        # if collection:
        #     response = collection.query.hybrid(
        #         query=query_template["query"],
        #         return_metadata=wvc.query.MetadataQuery(score=True, explain_score=True),
        #         fusion_type=weaviate.classes.query.HybridFusion.RELATIVE_SCORE,
        #         vector=query_template["query_vector"],
        #         limit=query_template["top_n"],
        #         alpha=query_template["alpha"],
        #     )
        #     print(response)              
        #     output = [obj.properties for obj in response.__dict__["objects"]]            
        #     return output            
        
        # logger.debug(f"{target_collection_name} does not exists!")
        # return None
        try:
            response = collection.query.hybrid(
                query=query_template["query"],
                return_metadata=wvc.query.MetadataQuery(score=True, explain_score=True),
                # fusion_type=weaviate.classes.query.HybridFusion.RELATIVE_SCORE,
                vector=query_template["query_vector"],
                limit=query_template["top_n"],
                alpha=query_template["alpha"],
            )

            if not response or len(response.objects) == 0:
                logging.info("No objects found in response.")
                return []
            
            results = []
            for obj in response.__dict__["objects"]:
                properties = obj.properties
                # metadata = obj.get("metadata", {})
                metadata = getattr(obj, "metadata", {})
                score = metadata.__dict__.get("score", 0.0)

                results.append(
                    {
                        "properties" : properties,
                        "score" : score,
                    }
                )
            return results
        
        except Exception as e:
            logging.error(f"Error during query execution: {str(e)}")
            return []
        
        
    # def get_collection_data(self, collection_name: str) -> List[Dict[str, Any]]:       
    #     """
    #     Fetches all data from a specified collection without relying on vectorization.
        
    #     :param collection_name: Name of the collection to retrieve data from.
    #     :return: A list of objects in the collection.
    #     """
    #     try:
    #         # Use a simple filter to fetch all objects
    #         response = self.get_collection(collection_name).query.all(limit=10000).do()
            
    #         if "data" in response and "Get" in response["data"] and collection_name in response["data"]["Get"]:
    #             return response["data"]["Get"][collection_name]
            
    #         return []
    #     except Exception as e:
    #         print(f"Error fetching collection data: {str(e)}")
    #         return []
    
    # class WeaviateClient:
    # def get_collection_data(self, collection_name: str) -> List[Dict[str, Any]]:
    #     """
    #     Fetches all data from a specified collection without relying on vectorization.
        
    #     :param collection_name: Name of the collection to retrieve data from.
    #     :return: A list of objects in the collection.
    #     """
    #     try:
    #         # Use a GraphQL query to fetch all objects
    #         query = {
    #             "query": f"""
    #             {{
    #                 Get {{
    #                     {collection_name} {{
    #                         source
    #                     }}
    #                 }}
    #             }}
    #             """
    #         }
            
    #         response = self.client.query.raw(query)

    #         # Parse response and extract data
    #         if "data" in response and "Get" in response["data"] and collection_name in response["data"]["Get"]:
    #             return response["data"]["Get"][collection_name]
            
    #         return []
    #     except Exception as e:
    #         print(f"Error fetching collection data: {str(e)}")
    #         return []
