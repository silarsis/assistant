from typing import Optional, Literal, Tuple, Union
from pydantic import BaseModel

from openai import AzureOpenAI, OpenAI

from langchain_openai import AzureChatOpenAI, ChatOpenAI

from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AsyncAzureOpenAI
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion


class LLMConnect(BaseModel):
    """
    Connect to any given LLM, whether via langchain, semantic kernel, or other
    This deals with various awkwardnesses like needing env variables set etc
    """
    api_type: Literal["openai", "azure"] = "openai"
    api_base: Optional[str] = None
    api_key: str = ""
    api_version: str = "2023-06-01-preview"
    deployment_name: str = "gpt-4"
    org_id: Optional[str] = None
    
    def langchain(self) -> Union[ChatOpenAI,AzureChatOpenAI]:
        " Connect via Langchain "
        if self.api_type == 'azure':
            llm = AzureChatOpenAI(client=self.openai(), model=self.deployment_name)
        else:
            llm = ChatOpenAI(client=self.openai(), model=self.deployment_name)
        return llm
    
    def sk(self) -> Tuple[str, Union[AzureChatCompletion, OpenAIChatCompletion]]:
        " Connect via Semantic Kernel "
        service_id = self.deployment_name
        if self.api_type == "azure":
            client = AsyncAzureOpenAI(api_key=self.api_key, organization=self.org_id, base_url=self.api_base)
            service = AzureChatCompletion(self.deployment_name, async_client=client, service_id=service_id)
        else:
            client = AsyncOpenAI(api_key=self.api_key, organization=self.org_id, base_url=self.api_base)
            service = OpenAIChatCompletion(service_id, async_client=client, service_id=service_id)
        return service_id, service
    
    def openai(self) -> Union[AzureOpenAI, OpenAI]:
        " Connect via the base OpenAI "
        if self.api_type == 'azure':
            # gets the API Key from environment variable AZURE_OPENAI_API_KEY
            client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                organization=self.org_id,
                base_url=self.api_base
            )
        else:
            client = OpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                organization=self.org_id,
                base_url=self.api_base
            )
        return client