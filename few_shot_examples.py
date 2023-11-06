from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from pydantic import Field, BaseModel

from utils import query_embeddings

# TODO: This is a modified version from the original few_shot_examples.py
FEW_SHOT_EXAMPLE_DESCRIPTION = """
    Input: question to find examples answers for.
    Output: One similar question's response.
    Use this tool to fetch previously asked Question/SQL pairs as examples for improving SQL query generation.
    """  # noqa: E501


class FewShotExampleInput(BaseModel):
    question: str = Field()


def few_shot_examples_func(string):
    """Get few-shot examples from the pool of samples."""
    return query_embeddings(string)


class FewShotExamplesTool(BaseTool):
    name = "few_shot_examples_tool"
    description = FEW_SHOT_EXAMPLE_DESCRIPTION
    args_schema: Type[BaseModel] = FewShotExampleInput

    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return few_shot_examples_func(query)

    async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
