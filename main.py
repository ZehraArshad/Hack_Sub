from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
import os
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_qdrant import Qdrant
# from langchain_community.vectorstores.qdrant import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType
from qdrant_client.models import KeywordIndexParams, KeywordIndexType
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === QDRANT SETUP ===
COLLECTION_NAME = "medpal-pdfs"
EMBEDDINGS = CohereEmbeddings(model="embed-english-v3.0")
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Create collection if it doesn't exist
if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

# Ensure payload index for user_id filtering
qdrant_client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="user_id",
    field_schema=KeywordIndexParams(
        type=KeywordIndexType.KEYWORD,
        is_tenant=True  # Optional, but helpful for multi-user filtering
    )
    
    )
# === ROUTES ===
@app.get("/")
def home():
    return {"message": "FastAPI is working!"}

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    print(f"📥 Received upload from user_id: {user_id}, file: {file.filename}")
    print(qdrant_client.get_collection(COLLECTION_NAME))
    print(qdrant_client.get_collection(COLLECTION_NAME).payload_schema)

    contents = await file.read()
    file_path = f"temp_{uuid4()}.pdf"

    # Save file temporarily
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        # Load and split PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)

        # Inject user_id into metadata
        for chunk in chunks:
            if not chunk.metadata:
                chunk.metadata = {}
            chunk.metadata["user_id"] = user_id

        # Store chunks in Qdrant
        Qdrant.from_documents(
            documents=chunks,
            embedding=EMBEDDINGS,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=COLLECTION_NAME,
            prefer_grpc=False
        )

        return {
            "message": "PDF uploaded and processed successfully",
            "chunks": len(chunks)
        }

    finally:
        os.remove(file_path)

# === CHAT ===
PROMPT_TEMPLATE = """
You are a helpful assistant. Use the context below to answer the question.
If the answer cannot be found in the context, reply based on your own knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

llm = init_chat_model(
    "llama3-8b-8192", model_provider="groq", api_key=os.getenv("GROQ_API_KEY")
)

class ChatRequest(BaseModel):
    question: str
    # user_id: str

@app.post("/chat")
async def chat(body: ChatRequest):
    query = body.question
    # user_id = body.user_id

    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=EMBEDDINGS,
    )

    # Filter using user_id
    docs = vectorstore.similarity_search(query, k=3, 
                                        #  removed filtering; causing index issues
                                         )

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt_messages = prompt.invoke({"context": context, "question": query})
    response = llm.invoke(prompt_messages)

    return {
        "answer": response.content,
        "chunks_used": len(docs)
    }
