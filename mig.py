from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import KeywordIndexParams, KeywordIndexType
from dotenv import load_dotenv
import os
load_dotenv()
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
if not client.collection_exists("medpal-pdfs"):
    client.create_collection(
        collection_name="medpal-pdfs",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    client.create_payload_index(
        collection_name="medpal-pdfs",
        field_name="user_id",
        field_schema=KeywordIndexParams(type=KeywordIndexType.KEYWORD, is_tenant=True),
    )
print("Collection is ready.")
