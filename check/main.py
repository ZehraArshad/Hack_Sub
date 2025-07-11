from fastapi import FastAPI
import uvicorn
from pyngrok import ngrok

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI via ngrok"}
