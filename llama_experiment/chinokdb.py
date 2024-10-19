from langchain.embeddings import OllamaEmbeddings
from langchain.llms import TextGen

from llama_index import (
    GPTKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    SQLDatabase,
    SQLStructStoreIndex, VectorStoreIndex,
)
from llama_index.indices.struct_store import SQLContextContainerBuilder, SQLTableRetrieverQueryEngine, \
    NLSQLTableQueryEngine
from llama_index.llms import Ollama
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
# Set up LLM
llm = Ollama(model="llama2", temperature=0.1)
llm_predictor = LLMPredictor(llm)
embed_model = OllamaEmbeddings(model="llama2")
'''
Define service integrated context, if not specified will use default settings:
    - llm_predictor: BaseLLMPredictor
    - prompt_helper: PromptHelper
    - embed_model: BaseEmbedding
    - node_parser: NodeParser
'''
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)

engine = create_engine("sqlite:///:memory:")
