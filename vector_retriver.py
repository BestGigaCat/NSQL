# Class for vector store retriver based on OllamaEmbeddings
from langchain.vectorstores import Chroma


def retrive_similar_questions(question: str, vectorstore: Chroma) -> list:
    docs = vectorstore.similarity_search(question)
    return docs
