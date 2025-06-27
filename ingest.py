from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
import os

load_dotenv()

PDF_FOLDER = "data"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "medpal-pdfs"

def load_and_split_pdfs():
    loader = DirectoryLoader(PDF_FOLDER, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"✅ Loaded {len(documents)} PDFs and split into {len(chunks)} chunks")
    return chunks


def save_to_qdrant(chunks):
    embeddings = CohereEmbeddings(model="embed-english-v3.0")

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print(f"✅ Created Qdrant collection '{COLLECTION_NAME}'")

    qdrant = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
    url=os.getenv("QDRANT_URL"),          
    prefer_grpc=False,                   
    api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=COLLECTION_NAME,
    )
    print(f"✅ Saved {len(chunks)} chunks to Qdrant Cloud")

def main():
    chunks = load_and_split_pdfs()
    save_to_qdrant(chunks)

if __name__ == "__main__":
    main()
