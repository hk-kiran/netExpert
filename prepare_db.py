from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

model_name = "llama3.2"

print("----------Creating Database--------------")


loader = PyPDFDirectoryLoader("data/")

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="netmanuals",
    embedding=OllamaEmbeddings(model=model_name),
    persist_directory="./chroma_langchain_db"
)
print(f"Total number of documents: {len(docs)}")
print("----------Created Database--------------")