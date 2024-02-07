import os

from pydantic import BaseModel
from semantic_kernel.plugin_definition import kernel_function

from langchain_community.utilities import GoogleSearchAPIWrapper

from config import settings

class GoogleSearchPlugin(BaseModel):
    @kernel_function(description="Scrape text from a website", name="search")
    async def search(self, query: str) -> str:
        """Use the Google Search API to search for and return results for the given query"""
        if settings.google_api_key:
            search = GoogleSearchAPIWrapper()
            results = search.run(query)
        else:
            results = "No Google API key found. Please set the GOOGLE_API_KEY environment variable."
        return str(results)