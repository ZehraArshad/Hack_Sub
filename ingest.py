# ingest.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

DATA_DIR = "data"             # Folder with PDFs
CHROMA_DIR = "chroma"         # Vector DB folder

def main():
    # Clear previous index (optional)
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    # Load all PDFs from folder
    all_docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(DATA_DIR, filename)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} PDF pages.")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks.")

    # Embedding model
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        input_type="classification",
        embedding_type="float"
    )

    # Save to Chroma
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    vectordb.persist()
    print(f"Saved to Chroma at {CHROMA_DIR}")

if __name__ == "__main__":
    main()
