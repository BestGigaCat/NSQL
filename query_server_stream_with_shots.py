from langchain.agents import initialize_agent, AgentType
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
from langchain_experimental.sql import SQLDatabaseChain

from CustomSQLChain import CustomSQLChain
import few_shot_examples
from agent_prompts import SUFFIX_WITH_FEW_SHOT_SAMPLES
from langchain.globals import set_debug
from langchain.agents import initialize_agent

set_debug(True)

model_url = "ws://192.168.0.28:5005"

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = TextGen(temperature=0, model_url=model_url, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

db_chain = CustomSQLChain.from_llm(llm, db, return_sql=True)
db_chain.run("How many employees are there?")

# tools = [
#     few_shot_examples.FewShotExamplesTool()
# ]

# agent = initialize_agent(
#     # tools,
#     llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False, max_iterations=3,
#     early_stopping_method='generate',
# )
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#
# agent.run("How to get the total number of employees in the table?")
