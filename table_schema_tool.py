from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from pydantic import Field, BaseModel

# from utils import query_embeddings

TABLE_SCHEMA_TOOL_DESCRIPTION = """
    Input: question to find examples answers for.
    Output: One table and column name separated by ->. Example output: table1 -> column2
    """  # noqa: E501


class TableSchemaInput(BaseModel):
    question: str = Field()


class TableSchemaTool(BaseTool):
    name = "table_schema_tool"
    description = TABLE_SCHEMA_TOOL_DESCRIPTION
    args_schema: Type[BaseModel] = TableSchemaInput

    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return "Employee -> *"

    async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
