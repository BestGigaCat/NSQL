from langchain.llms import TextGen

from llama_index import (
    GPTKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    SQLDatabase,
    SQLStructStoreIndex, VectorStoreIndex,
)
from llama_index.indices.struct_store import SQLContextContainerBuilder, SQLTableRetrieverQueryEngine
from llama_index.objects import SQLTableNodeMapping, SQLTableSchema, ObjectIndex

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
)

from sqlalchemy import insert

# Define custom LLM
# https://gpt-index.readthedocs.io/en/v0.6.6/how_to/customization/custom_llms.html
model_url = "http://localhost:5000"
# https://python.langchain.com/docs/integrations/llms/textgen
# Consider configure temperature
llm = TextGen(model_url)
llm_predictor = LLMPredictor(llm)

'''
Define service integrated context, if not specified will use default settings:
    - llm_predictor: BaseLLMPredictor
    - prompt_helper: PromptHelper
    - embed_model: BaseEmbedding
    - node_parser: NodeParser
'''
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# Connect database
engine = create_engine("sqlite:///:memory:")

# Set up metadata in memory store
# Holds a collection of :class:`_schema.Table` objects
metadata_obj = MetaData()
