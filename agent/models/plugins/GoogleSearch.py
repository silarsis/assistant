import os

from pydantic import BaseModel
from semantic_kernel.functions.kernel_function_decorator import kernel_function

from langchain_community.utilities import GoogleSearchAPIWrapper

from config import settings

class GoogleSearchPlugin(BaseModel):
    @kernel_function(description="Use google search API to return answers for a query, takes a query and returns the scraped text from relevant websites", name="search")
    async def search(self, query: str) -> str:
        """Use the Google Search API to search for and return results for the given query"""
        if settings.google_api_key:
            search = GoogleSearchAPIWrapper()
            results = search.run(query)
        else:
            results = "No Google API key found. Please set the GOOGLE_API_KEY environment variable."
        return str(results)