from pydantic import BaseModel, Field

from utils import query_embeddings

# TODO: This is a modified version from the original few_shot_examples.py
FEW_SHOT_EXAMPLE_DESCRIPTION = """
    Input: question to find examples answers for.
    Output: One similar question's response.
    Use this tool to fetch previously asked Question/SQL pairs as examples for improving SQL query generation.
    """  # noqa: E501


class FewShotExampleInput(BaseModel):
    question: str = Field()


def few_shot_examples_func(question: str) -> str:
    """Get few-shot examples from the pool of samples."""
    return query_embeddings("playground2")
