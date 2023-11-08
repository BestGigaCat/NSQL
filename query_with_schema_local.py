from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import TextGen
from agent_prompts import SUFFIX_WITH_FEW_SHOT_SAMPLES
from table_schema_tool import TableSchemaTool

model_url = "ws://192.168.0.28:5005"
llm = TextGen(model_url=model_url,  streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
prompt_template = (SUFFIX_WITH_FEW_SHOT_SAMPLES + "You are an SQL expert. Can you generate SQL for {question}? \n"

                                                  "You can use table_schema_tool to get examples first\n"
                   )

prompt = PromptTemplate(
    input_variables=["question", "tool_names"], template=prompt_template
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain)

# Replace tool here for different testings
tools = [TableSchemaTool()]
executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, handle_parsing_errors=True)

print("ask")

executor.run({"question": "How many employees are there", "tool_names": ["table_schema_tool"]})
