from pydantic import BaseModel
from typing import Annotated

import openai
from models.tools.llm_connect import LLMConnect
from semantic_kernel.functions.kernel_function_decorator import kernel_function

from config import settings

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
    if settings.img_openai_inherit:
        api_key: str = settings.openai_api_key
        base_url: str = settings.openai_api_base
        api_version: str = settings.openai_api_version
        api_type: str = settings.openai_api_type
        org_id: str = settings.openai_org_id
    else:
        api_key: str = settings.img_openai_api_key
        base_url: str = settings.img_openai_api_base
        api_version: str = settings.img_openai_api_version
        api_type: str = settings.img_openai_api_type
        org_id: str = settings.img_openai_org_id
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @kernel_function(name="gen_image", description="Generate an image from a description")
    def gen_image(self, description: Annotated[str, "The image description"] = "") -> str:
        " Generate an image from a description "
        client = LLMConnect(
            api_type=self.api_type, 
            api_key=self.api_key, 
            api_base=self.base_url, 
            org_id=self.org_id
        ).openai()
        result = client.images.generate(
            prompt=description, 
            size="1024x1024",
            n=1)
        return f'<img src="{result.data[0].url}" alt="{description}" style="max-width: 50%; max-height: 50%;">'