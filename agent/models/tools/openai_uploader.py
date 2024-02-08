import base64
import requests

from config import settings


def upload_image(image_path):
    # Getting the base64 string
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {settings.openai_api_key}"
    }

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Please describe this image"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    # This should be async
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()