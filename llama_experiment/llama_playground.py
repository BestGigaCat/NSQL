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
sql_database = SQLDatabase(engine, include_tables=["city_stats"])

# https://gpt-index.readthedocs.io/en/latest/examples/index_structs/struct_indices/SQLIndexDemo.html
# Set sample data for testing
# TODO: replace with local Chinook DB
sql_database = SQLDatabase(engine, include_tables=["city_stats"])
table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)
rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {
        "city_name": "Chicago",
        "population": 2679000,
        "country": "United States",
    },
    {"city_name": "Seoul", "population": 9776000, "country": "South Korea"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

# If the db is already populated with data, we can instantiate the SQL index with a blank documents list Note: this
# index only covers one table, for many table index, check out:
# https://gpt-index.readthedocs.io/en/v0.6.27/examples/index_structs/struct_indices/SQLIndexDemo-ManyTables.html
index = SQLStructStoreIndex(
    [],
    sql_database=sql_database,
    table_name="city_stats",
)

# view current table
stmt = select(
    city_stats_table.c["city_name", "population", "country"]
).select_from(city_stats_table)

with engine.connect() as connection:
    results = connection.execute(stmt).fetchall()
    print(results)

# Ingesting context for tables in DB
# manually set text
# https://gpt-index.readthedocs.io/en/v0.6.27/guides/tutorials/sql_guide.html
# This is set for one specific table
# This is for from manual input, we can also extract context from documents
# This is mostly used for table descriptions
city_stats_text = (
    "This table gives information regarding the population and country of a given city.\n"
    "The user will query with codewords, where 'foo' corresponds to population and 'bar'"
    "corresponds to city."
)
table_context_dict = {"city_stats": city_stats_text}
context_builder = SQLContextContainerBuilder(sql_database, context_dict=table_context_dict)
context_container = context_builder.build_context_container()

# building the index
# TODO: this duplicate with the previous SQLStructStoreIndex above, remove one of them
index = SQLStructStoreIndex(
    [],
    sql_database=sql_database,
    table_name="city_stats",
    sql_context_container=context_container
)

# Multi-table index set up
# create a ton of dummy tables
n = 100
all_table_names = ["city_stats"]
for i in range(n):
    tmp_table_name = f"tmp_table_{i}"
    tmp_table = Table(
        tmp_table_name,
        metadata_obj,
        Column(f"tmp_field_{i}_1", String(16), primary_key=True),
        Column(f"tmp_field_{i}_2", Integer),
        Column(f"tmp_field_{i}_3", String(16), nullable=False),
    )
    all_table_names.append(f"tmp_table_{i}")

metadata_obj.create_all(engine)

# print tables
metadata_obj.tables.keys()

# build a vector index from the table schema information
context_builder = SQLContextContainerBuilder(sql_database)
table_schema_index = context_builder.derive_index_from_context(
    VectorStoreIndex,
)

# Send a sample query
query_str = "Which city has the highest population?"
query_engine = index.as_query_engine(
    sql_context_container=context_container
)
response = query_engine.query(query_str)


# Query for if you don't know which table to use
# https://gpt-index.readthedocs.io/en/latest/examples/index_structs/struct_indices/SQLIndexDemo.html#
# set Logging to DEBUG for more detailed outputs
table_node_mapping = SQLTableNodeMapping(sql_database)
# TODO: pass in all schemas
table_schema_objs = [
    (SQLTableSchema(table_name="city_stats"))
]  # add a SQLTableSchema for each table

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
query_engine = SQLTableRetrieverQueryEngine(
    sql_database, obj_index.as_retriever(similarity_top_k=1)
)


