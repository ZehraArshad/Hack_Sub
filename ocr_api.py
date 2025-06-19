import requests
import os

def call_ocr_space_api(file_path: str, api_key: str):
    url = "https://api.ocr.space/parse/image"
    payload = {
        "language": "eng",
        "isOverlayRequired": False,
    }
    with open(file_path, "rb") as f:
        response = requests.post(
            url,
            data=payload,
            files={"file": f},
            headers={"apikey": api_key}
        )
    result = response.json()
    parsed_text = result["ParsedResults"][0]["ParsedText"] if "ParsedResults" in result else None
    return parsed_text
