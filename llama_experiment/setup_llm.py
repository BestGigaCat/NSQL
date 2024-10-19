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

# Connect database
engine = create_engine("sqlite:///:memory:")

# Set up metadata in memory store
# Holds a collection of :class:`_schema.Table` objects
metadata_obj = MetaData()

# create city SQL table
table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)
metadata_obj.create_all(engine)

# https://gpt-index.readthedocs.io/en/latest/examples/index_structs/struct_indices/SQLIndexDemo.html
# Set sample data for testing
# TODO: replace with local Chinook DB
rows = [
    {"city_name": "Toronto", "population": 2731571, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13929286, "country": "Japan"},
    {"city_name": "Berlin", "population": 600000, "country": "Germany"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

# view current table
stmt = select(
    city_stats_table.c.city_name,
    city_stats_table.c.population,
    city_stats_table.c.country,
).select_from(city_stats_table)

with engine.connect() as connection:
    results = connection.execute(stmt).fetchall()
    print(results)

sql_database = SQLDatabase(engine, include_tables=["city_stats"])
# If the db is already populated with data, we can instantiate the SQL index with a blank documents list Note: this
# index only covers one table, for many table index, check out:
# https://gpt-index.readthedocs.io/en/v0.6.27/examples/index_structs/struct_indices/SQLIndexDemo-ManyTables.html
# index = SQLStructStoreIndex(
#    [],
#    sql_database=sql_database,
#    table_name="city_stats",
#    service_context=service_context
#)

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["city_stats"],
    service_context=service_context
)
query_str = "Which city has the highest population?"
response = query_engine.query(query_str)
print(response)

