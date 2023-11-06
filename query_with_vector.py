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
from langchain.tools import Tool
from langchain.utilities import SQLDatabase

from few_shot_examples import few_shot_examples_func, FewShotExampleInput

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Set up LLM
llm = Ollama(
    model="llama2",
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

prompt_template = ("Please generate SQL for {question}, use few_shot_examples_tool to get examples first")
prompt = PromptTemplate(
    input_variables=["question"], template=prompt_template
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain)
tools = []
tools.append(
    Tool.from_function(
        func=few_shot_examples_func,
        name="few_shot_examples_tool",
        description="FEW_SHOT_EXAMPLE_DESCRIPTION",
        args_schema=FewShotExampleInput,
    )
)
executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools)

print("ask")

response = executor.run({"question": "How many employees are there"})
