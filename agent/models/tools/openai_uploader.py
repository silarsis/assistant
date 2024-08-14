import base64
import requests
import json

from config import settings

# Should be able to work this into the graph now


def upload_image(image_path: str) -> str:
    # Getting the base64 string
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {settings.img_upload_api_key or settings.openai_api_key}"
    }

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Please describe this image in as much detail as necessary to be able to accurately recreate it from just the description. If this is a diagram, be sure to capture all aspects of the diagram. If it's a picture, be as descriptive as possible."
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
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
    try:
        return response.json()['choices'][0]['message']['content']
    except (IndexError, KeyError) as e:
        print(f"Error uploading image: {e}")
        return json.loads(response.text)['error']['message']