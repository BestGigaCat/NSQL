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
