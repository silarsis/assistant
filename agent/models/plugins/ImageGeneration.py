from pydantic import BaseModel
import os

import openai
from semantic_kernel.plugin_definition import kernel_function, kernel_function_context_parameter

# This is a hack to access Dall-E-2 in US East, because I don't have Dall-E-3 in my resource currently.
import time
import json
import httpx

class CustomHTTPTransport(httpx.HTTPTransport):
    def handle_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        if "images/generations" in request.url.path and request.url.params[
            "api-version"
        ] in [
            "2023-06-01-preview",
            "2023-07-01-preview",
            "2023-08-01-preview",
            "2023-09-01-preview",
            "2023-10-01-preview",
        ]:
            request.url = request.url.copy_with(path="/openai/images/generations:submit")
            response = super().handle_request(request)
            if response.status_code not in (200, 202):
                raise httpx.HTTPError(response=response)
            operation_location_url = response.headers["operation-location"]
            request.url = httpx.URL(operation_location_url)
            request.method = "GET"
            response = super().handle_request(request)
            response.read()

            timeout_secs: int = 120
            start_time = time.time()
            while response.json()["status"] not in ["succeeded", "failed"]:
                if time.time() - start_time > timeout_secs:
                    timeout = {"error": {"code": "Timeout", "message": "Operation polling timed out."}}
                    return httpx.Response(
                        status_code=400,
                        headers=response.headers,
                        content=json.dumps(timeout).encode("utf-8"),
                        request=request,
                    )

                time.sleep(int(response.headers.get("retry-after", 10)) or 10)
                response = super().handle_request(request)
                response.read()

            if response.json()["status"] == "failed":
                error_data = response.json()
                return httpx.Response(
                    status_code=400,
                    headers=response.headers,
                    content=json.dumps(error_data).encode("utf-8"),
                    request=request,
                )

            result = response.json()["result"]
            return httpx.Response(
                status_code=200,
                headers=response.headers,
                content=json.dumps(result).encode("utf-8"),
                request=request,
            )
        return super().handle_request(request)

class ImageGenerationPlugin(BaseModel):
    api_key: str = os.environ.get("IMG_OPENAI_API_KEY", "")
    base_url: str = os.environ.get("IMG_OPENAI_API_BASE", "")
    api_version: str = os.environ.get("IMG_OPENAI_API_VERSION", "")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @kernel_function(
        description="Generate an image from a description",
        name="generate_image",
        input_description="Description of the image"
    )
    @kernel_function_context_parameter(
        name="description",
        description="The image description"
    )
    def gen_image(self, description: str = "") -> str:
        client = openai.AzureOpenAI(
            api_key=self.api_key, 
            azure_endpoint=self.base_url, 
            api_version=self.api_version,
            http_client=httpx.Client(
                transport=CustomHTTPTransport(),
            )
        )
        result = client.images.generate(
            prompt=description, 
            size="1024x1024",
            n=1)
        return f'<img src="{result.data[0].url}" alt="{description}" style="max-width: 50%; max-height: 50%;">'