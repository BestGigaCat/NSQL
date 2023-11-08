from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
# QA chain
# from chromadb.config import Settings
# import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.utilities import SQLDatabase
from langchain.llms import TextGen
from table_schema_tool import TableSchemaTool

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Set up LLM
llm = Ollama(
    model="llama2",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

prompt_template = ("You are an SQL expert. Please generate SQL for {question}, \n"
                   "1. use table_schema_tool to get a table to consider answering this question\n"
                   "2. then tell me what table you decide to use\n"
                   "3. Then tell me the SQL query you generate for this {question}\n")
prompt = PromptTemplate(
    input_variables=["question"], template=prompt_template
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain)

# Replace tool here for different testings
tools = [TableSchemaTool()]
executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools)

print("ask")

response = executor.run({"question": "How many employees are there"})
