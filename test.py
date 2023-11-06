# Load web page
import datetime

from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Embed and store
from langchain.vectorstores import Chroma

print("start")
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

print("split")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
all_splits = text_splitter.split_documents(data)

print("load")
embed = OllamaEmbeddings(base_url="http://localhost:11434", model="llama2")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed, persist_directory="./chroma_db")

print("ask")
# Retrieve
question = "How can Task Decomposition be done?"
docs = vectorstore.similarity_search(question)
print("end")
print(len(docs))

# print the current date and time
print("Current time is: ", datetime.now())
