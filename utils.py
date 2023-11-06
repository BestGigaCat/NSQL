import chromadb
from langchain.embeddings import OllamaEmbeddings


def query_embeddings(question):
    print("Loading")
    persistent_client = chromadb.PersistentClient("./chroma_db")
    collection = persistent_client.get_or_create_collection(name="playground2")

    # Set up embedding
    print("Generate embedings")
    embed = OllamaEmbeddings(base_url="http://localhost:11434", model="llama2")
    embeddings = embed.embed_query(text=question)

    print("Search docs")
    docs = collection.query(query_embeddings=embeddings)
    print(len(docs))
    print(docs)
    return docs[0]["metadatas"][0]["query"]
