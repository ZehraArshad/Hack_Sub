from fastapi import FastAPI, UploadFile, File
from ocr_api import call_ocr_space_api
import os
import shutil
from dotenv import load_dotenv


load_dotenv()
OCR_API_KEY = os.getenv("OCR_API_KEY")
  # Replace with your real API key

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        text = call_ocr_space_api(file_path, OCR_API_KEY)
    finally:
        os.remove(file_path)  # Clean up

    return {"extracted_text": text or "No text found"}


# query through grok