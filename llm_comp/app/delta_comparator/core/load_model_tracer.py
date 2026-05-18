"""Loads LLM model."""
import os
from langchain_openai import OpenAI
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from app.delta_comparator.utils.logger import log as logging

    
# class TokenUsageHandler(BaseCallbackHandler):
#     """Custom callback handler to track token usage only from raw LLM traces."""

#     def __init__(self):
#         self.total_prompt_tokens = 0
#         self.total_completion_tokens = 0
#         self.total_tokens = 0

#     def on_llm_end(self, response, **kwargs):
#         try:
#             usage = None

#             # Raw LLM response usually contains llm_output
#             if hasattr(response, "llm_output") and response.llm_output:
#                 usage = response.llm_output.get("usage")

#             # Some providers (Azure, Databricks) tuck it under additional_kwargs
#             if not usage and hasattr(response, "additional_kwargs"):
#                 usage = response.additional_kwargs.get("usage")

#             if usage:
#                 self.total_prompt_tokens += usage.get("prompt_tokens", 0)
#                 self.total_completion_tokens += usage.get("completion_tokens", 0)
#                 self.total_tokens += usage.get("total_tokens", 0)

#                 loguru.logger.info(f"Token usage captured: {usage}")
#             else:
#                 loguru.logger.debug("No token usage in this trace.")
#         except Exception as e:
#             loguru.logger.warning(f"Token tracking failed: {e}")

#     def get_usage(self):
#         return {
#             "prompt_tokens": self.total_prompt_tokens,
#             "completion_tokens": self.total_completion_tokens,
#             "total_tokens": self.total_tokens,
#         }

class LLMModel:
    """Load LLM Model."""

    def __init__(self):
        """Initialize the class."""
        self.env = os.getenv("CHANGE_TRACER_ENV", "vllm").lower()
        #self.token_handler = TokenUsageHandler()        

    def load_model(self):
        """Placeholder to load the model based on env variables.

        Args:
            env (pd.Series): CHANGE_TRACER_ENV.

        Returns:
            Model to load.
        """
        if self.env == "azure":
            #logging.debug("Using Async Azure OpenAI")
            return self.AzureChatOpenAI()
        elif self.env == "databricks":
            #logging.debug("Using Databricks ChatOpenAI")
            return self.DatabricksChatOpenAI()
        elif self.env == "vllm":
            #logging.debug("Using Langchain ChatOpenAI")
            return self.VllmOpenAI()
        else:
            raise ValueError(f"Unsupported environment: {self.env}")

    # # --- Token usage API ---
    # def get_token_usage(self):
    #     return {
    #         "prompt_tokens": self.token_handler.prompt_tokens,
    #         "completion_tokens": self.token_handler.completion_tokens,
    #         "total_tokens": self.token_handler.total_tokens,
    #     }

    # def reset_token_usage(self):
    #     self.token_handler.reset()

    # CHANGE TRACER API END-POINTS
    def VllmOpenAI(self):
        """Loads VLLM Model."""
        llm = OpenAI(
            model=os.getenv("VLLM_MODEL_NAME"),
            temperature=float(os.getenv("VLLM_MODEL_TEMPERATURE")),
            api_key=os.getenv("VLLM_PASSWORD"),
            base_url=os.getenv("VLLM_ENDPOINT"),
            max_tokens=int(os.getenv("TRACER_MAX_TOKENS")),
            streaming=bool(os.getenv("TRACER_IS_STREAMING")),
        )
        return llm

    def AzureChatOpenAI(self):
        """Loads Azure Chat OpenAI."""
        llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_version=os.getenv("API_VERSION"),
            model=os.getenv("AZURE_MODEL"),
            temperature=float(os.getenv("VLLM_MODEL_TEMPERATURE")),
            max_tokens=int(os.getenv("TRACER_MAX_TOKENS")),
            streaming=bool(os.getenv("TRACER_IS_STREAMING")),
            #callbacks=[self.token_handler],
        )
        return llm

    def DatabricksChatOpenAI(self):
        """Loads Databricks Chat OpenAI."""
        llm = ChatOpenAI(
            api_key=os.getenv("DATABRICKS_KEY"),
            base_url=os.getenv("DATABRICKS_ENDPOINT"),
            model=os.getenv("DATABRICKS_MODEL"),
            temperature=float(os.getenv("VLLM_MODEL_TEMPERATURE")),
            max_tokens=int(os.getenv("TRACER_MAX_TOKENS")),
            streaming=bool(os.getenv("TRACER_IS_STREAMING")),
        )
        return llm

    # --- Usage accessor ---
    # def get_token_usage(self):
    #     return self.token_handler.get_usage()