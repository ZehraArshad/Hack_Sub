from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from pydantic import BaseModel



load_dotenv()

app = FastAPI()

# Optional CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init MongoDB
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["medpal"]
reports_collection = db["reports"]

# Init Qdrant
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = "medpal-pdfs"
EMBEDDINGS = CohereEmbeddings(model="embed-english-v3.0")

# Create collection if it doesn't exist
if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )


@app.get("/")
def home():
    return {"message": "FastAPI is working!"}


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Form(...)  # for now, use "123"
):
    contents = await file.read()
    file_path = f"temp_{uuid4()}.pdf"

    # Save temp file
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        # Extract + split text
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)

        # Create a report entry
        report_id = str(uuid4())
        # reports_collection.insert_one({
        #     "report_id": report_id,
        #     "user_id": user_id,
        #     "filename": file.filename,
        #     "chunk_count": len(chunks)
        # })

        # Store in Qdrant
        qdrant = Qdrant.from_documents(
    documents=chunks,
    embedding=EMBEDDINGS,
    url=os.getenv("QDRANT_URL"),
    prefer_grpc=False,
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name=COLLECTION_NAME,
    metadata={"user_id": user_id, "report_id": report_id}
)   

        return {
            "message": "PDF uploaded and processed successfully",
            "report_id": report_id,
            "chunks": len(chunks)
        }

    finally:
        os.remove(file_path)
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

# Groq LLM
llm = init_chat_model(
    "llama3-8b-8192", model_provider="groq", api_key=os.getenv("GROQ_API_KEY")
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(body: ChatRequest):
    
    query = body.question
    # user_id = data.get("user_id")

    # if not query or not user_id:
    #     return {"error": "query and user_id are required."}

    # Filter by user_id if you stored that in metadata
    vectorstore = Qdrant(
        client=qdrant_client,   
        collection_name=COLLECTION_NAME,
        embeddings=EMBEDDINGS,
    )

    docs = vectorstore.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt_messages = prompt.invoke({"context": context, "question": query})
    response = llm.invoke(prompt_messages)

    return {
        "answer": response.content,
        "chunks_used": len(docs),
    }