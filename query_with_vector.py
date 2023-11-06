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

import few_shot_examples
from agent_prompts import SUFFIX_WITH_FEW_SHOT_SAMPLES

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Set up LLM
llm = Ollama(
    model="llama2",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

prompt_template = (SUFFIX_WITH_FEW_SHOT_SAMPLES + "You are an SQL expert. Can you generate SQL for {question}? \n"

                                                  "You can use few_shot_examples_tool to get examples first\n"
                   )

prompt = PromptTemplate(
    input_variables=["question", "tool_names"], template=prompt_template
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain)
tools = [
    few_shot_examples.FewShotExamplesTool()
]
executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools)

print("ask")

executor.run({"question": "How many employees are there", "tool_names": "few_shot_examples_tool"})
