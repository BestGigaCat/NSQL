import chromadb

# from chromadb.config import Settings

from langchain.embeddings import OllamaEmbeddings

# from langchain.vectorstores import Chroma

persistent_client = chromadb.PersistentClient("./chroma_db")
# collection = client.create_collection("lilianblog")
collection = persistent_client.get_or_create_collection(name="langchain")

print(persistent_client.list_collections())
print(collection.count())

# Generating Embeddings
print("Generate embedings")
embed = OllamaEmbeddings(base_url="http://localhost:11434", model="llama2")
embeddings = embed.embed_query(text="How can Task Decomposition be done?")

# Search docs 1
print("Search docs")
docs = collection.query(query_embeddings=embeddings)
print(len(docs))
print(docs)
