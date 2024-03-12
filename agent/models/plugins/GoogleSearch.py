from pydantic import BaseModel
from semantic_kernel.functions.kernel_function_decorator import kernel_function

from googleapiclient.discovery import build
from search_engine_parser.core.engines.google import Search as GoogleSearch

from config import settings

MAX_RESULTS = 10

class GoogleSearchPlugin(BaseModel):
    @kernel_function(description="Use google search API to return answers for a query, takes a query and returns the scraped text from relevant websites", name="search")
    async def search(self, query: str) -> str:
        """Use the Google Search API to search for and return results for the given query"""
        if settings.google_api_key:
            print("Searching via API")
            service = build("customsearch", "v1", developerKey=settings.google_api_key)
            cse = service.cse()
            results = cse.list(q=query, cx=settings.google_cse_id, num=MAX_RESULTS).execute().get("items", [])
            formatted_results = '\n'.join([item['snippet'] for item in results]) + '\n\nREFERENCES:\n' + '\n'.join([item['link'] for item in results])
        else: # Fallback to scraping the search results
            print("Searching via scraping")
            gsearch = GoogleSearch()
            resObj = await gsearch.async_search(query, MAX_RESULTS)
            results = resObj.results
            formatted_results = '\n'.join([f"{item['titles']} - {item['descriptions']}" for item in results]) + '\n\nREFERENCES:\n' + '\n'.join([item['links'] for item in results])
        return formatted_results
    
    def _convert_results_to_response(self, results: dict) -> str:
        return '\n'.join([item['snippet'] for item in results]) + '\n\nREFERENCES:\n' + '\n'.join([item['link'] for item in results])