# from chromadb.config import Settings
import chromadb
from langchain.embeddings import OllamaEmbeddings

# from langchain.vectorstores import Chroma


print("Loading")
persistent_client = chromadb.PersistentClient("./chroma_db")
# collection = client.create_collection("lilianblog")
collection = persistent_client.get_or_create_collection(name="playground2")
print(persistent_client.list_collections())

collection.delete(ids=["1"])

# Set up embedding
embed = OllamaEmbeddings(base_url="http://localhost:11434", model="llama2")
embeddings1 = embed.embed_query(text="How I get employee's title?")

# Adding to collection
collection.add(ids=["2"], embeddings=[embeddings1], metadatas=[{"query": "SELECT count(*) from Employee;"}])

# Generating Embeddings for golden query
print("Generate embedings")
question = "Can you generate a SQL script to get the total number of employees?"
embeddings2 = embed.embed_query(text=question)

# Search query
print("Search docs")
docs = collection.query(query_embeddings=embeddings2)
print(len(docs))
print(docs)
