from semantic_kernel.sk_pydantic import SKBaseModel
from semantic_kernel.plugin_definition import kernel_function

from models.tools.web_requests import scrape_text

class ScrapeTextPlugin(SKBaseModel):
    @kernel_function(description="Scrape text from a website", name="scrape_text")
    async def scrape_text(self, url: str) -> str:
        """Scrape text from a website"""
        return scrape_text(url)