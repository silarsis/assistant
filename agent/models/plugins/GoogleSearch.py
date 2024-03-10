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
            results = search.results(query, 10)
        else:
            results = "No Google API key found. Please set the GOOGLE_API_KEY environment variable."
        self._convert_results_to_response(results)
        return str(results)
    
    def _convert_results_to_response(self, results: dict) -> str:
        return '\n'.join([item['snippet'] for item in results]) + '\n\nREFERENCES:\n' + '\n'.join([item['link'] for item in results])