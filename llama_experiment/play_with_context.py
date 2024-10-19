from langchain.embeddings import OllamaEmbeddings
from langchain.llms import TextGen

from llama_index import (
    GPTKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    SQLDatabase,
    SQLStructStoreIndex, VectorStoreIndex, set_global_service_context
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
    inspect,
)

from sqlalchemy import insert

import os

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ["REPLICATE_API_TOKEN"] = "REPLICATE_API_TOKEN"

# currently needed for notebooks
import openai


import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)

from IPython.display import Markdown, display


llm = Ollama(model="llama2", temperature=0.01)
llm_predictor = LLMPredictor(llm)
embed_model = OllamaEmbeddings(model="llama2")
ctx = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)
set_global_service_context(ctx)

engine = create_engine("sqlite:///Chinook.db")
insp = inspect(engine)
db_list = insp.get_table_names()
print(db_list)
sql_database = SQLDatabase(engine)


# set Logging to DEBUG for more detailed outputs
table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    (SQLTableSchema(table_name="Album"), SQLTableSchema(table_name="Employee"), SQLTableSchema(table_name="Customer"))
]  # add a SQLTableSchema for each table

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
query_engine = SQLTableRetrieverQueryEngine(
    sql_database, obj_index.as_retriever(similarity_top_k=1)
)

response = query_engine.query("How many employees we have in total?")
display(Markdown(f"<b>{response}</b>"))

