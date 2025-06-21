from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

CHROMA_PATH = "chroma"
PDF_FOLDER = "data"

def load_and_split_pdfs():
    loader = DirectoryLoader(PDF_FOLDER, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"✅ Loaded {len(documents)} PDFs and split into {len(chunks)} chunks")
    return chunks

def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = CohereEmbeddings(
        model="embed-english-v3.0"
    )

    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    print(f"✅ Saved {len(chunks)} chunks to Chroma at '{CHROMA_PATH}'")

def main():
    chunks = load_and_split_pdfs()
    save_to_chroma(chunks)

if __name__ == "__main__":
    main()
