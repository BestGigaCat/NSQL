from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.utilities import SQLDatabase
from langchain.vectorstores import Chroma
from langchain_experimental.sql import SQLDatabaseChain

# Set up the baseLLM
llm = Ollama(
    verbose=True, model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

# Set up vector store and its retriever
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="llama2")
vectorstore = Chroma("langchain_store", oembed)
retriever = Chroma.as_retriever()

# Connect to local SQLite
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# Set up DB chain to enable LLm queries
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
