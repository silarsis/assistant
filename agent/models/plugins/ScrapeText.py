from pydantic import BaseModel
from semantic_kernel.functions.kernel_function_decorator import kernel_function

from models.tools.web_requests import scrape_text

class ScrapeTextPlugin(BaseModel):
    @kernel_function(description="Scrape text from a website, takes a URL and returns the scraped text", name="scrape_text")
    async def scrape_text(self, url: str) -> str:
        """Scrape text from a website"""
        return scrape_text(url)