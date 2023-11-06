from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama


def multiplier(a, b):
    return a * b


def parsing_multiplier(string):
    a, b = string.split(",")
    return multiplier(int(a), int(b))


llm = Ollama(
    model="llama2",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
tools = [
    Tool(
        name="Multiplier",
        func=parsing_multiplier,
        description="useful for when you need to multiply two numbers together. The input to this tool should be a comma separated list of numbers of length two, representing the two numbers you want to multiply together. For example, `1,2` would be the input if you wanted to multiply 1 by 2.",
    )
]
mrkl = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
mrkl.run("What is 3 times 4")
