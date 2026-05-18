"""Prompt for the output fixer."""
from langchain_core.prompts.prompt import PromptTemplate

CUSTOM_FIX = """Instructions:
--------------
The output provided does not conform to the required format or schema. Correct the output as follows:
1. The response must be a **valid JSON object**.
2. Enclose the entire JSON response in triple backticks (```), e.g., ```Output```.
3. **Do not include any text** outside the JSON object or explanatory comments.
--------------
Completion:
--------------
{completion}
--------------

Above, the Completion did not satisfy the constraints given in the Instructions.
Error:
--------------
{error}
--------------

REQUIRED ACTION:
--------------
Fix the response to follow the above instructions. Your response must fully comply with the instructions. Provide only the corrected JSON object enclosed in triple backticks."""

CUSTOM_FIX_PROMPT = PromptTemplate.from_template(CUSTOM_FIX)
