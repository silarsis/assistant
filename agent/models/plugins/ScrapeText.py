from semantic_kernel.sk_pydantic import SKBaseModel
from semantic_kernel.skill_definition import sk_function

from models.tools.web_requests import scrape_text

class ScrapeTextSkill(SKBaseModel):
    @sk_function(description="Scrape text from a website", name="scrape_text")
    async def scrape_text(self, url: str) -> str:
        """Scrape text from a website"""
        return scrape_text(url)