SUFFIX_WITH_FEW_SHOT_SAMPLES = """Begin!
Use the following format:\n

Question: the input question you must answer\n
Thought: you should always think about what to do\n
Action: the action to take, should be one of [{tool_names}]\n
Action Input: the input to the action\n
Observation: the result of the action\n
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer\n
Final Answer: the final answer to the original input question\n

Question: {question}\n
Thought: I should Collect examples of Question/SQL pairs to identify possibly relevant tables, columns, and SQL query styles..\n
{agent_scratchpad}\n"""  # noqa: E501
