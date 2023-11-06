from langchain.tools.base import BaseTool
from pydantic import BaseModel, Extra


class BaseSQLDatabaseTool(BaseModel):
    """Base tool for interacting with the SQL database and the context information."""

    # db: SQLDatabase = Field(exclude=True)
    # context: List[dict] | None = Field(exclude=True, default=None)

    class Config(BaseTool.Config):
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        extra = Extra.forbid
