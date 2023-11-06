from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain

"""
The answer looks like this:

To retrieve the number of employees in a database, you can use a query like this:
```
SELECT COUNT(*) AS num_employees
FROM employees;
```
This will return the total number of rows in the `employees` table.

If you want to filter the results by a specific column, you can add a WHERE clause to the query. For example, to get the number of employees in each department, you can use a query like this:
```
SELECT departments.name AS department_name, COUNT(*) AS num_employees
FROM employees
JOIN departments ON employees.department = departments.id
GROUP BY departments.name;
```
This will return the number of employees in each department, grouped by department name

"""

# QA chain
# from chromadb.config import Settings
# import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Set up LLM
llm = Ollama(
    model="llama2",
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

prompt_template = ("Please generate SQL for {question}")
prompt = PromptTemplate(
    input_variables=["question"], template=prompt_template
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain)
executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=[])

print("ask")

response = executor.run({"question": "How many employees are there"})
