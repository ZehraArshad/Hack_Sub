from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.getenv("MONGODB_URI")

try:
    client = MongoClient(uri)
    dbs = client.list_database_names()
    print("✅ Connected to MongoDB. Databases:", dbs)
except Exception as e:
    print("❌ Connection failed:", e)
