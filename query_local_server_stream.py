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
from langchain.globals import set_debug

set_debug(True)

# Set up LLM
model_url = "ws://192.168.0.28:5005"
template = """Question: {question}
Answer: Let's think step by step."""


prompt = PromptTemplate(template=template, input_variables=["question"])
llm = TextGen(model_url=model_url, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain.run(question)
