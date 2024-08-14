from typing import Optional, Literal, Tuple, Union
from pydantic import BaseModel

from openai import AzureOpenAI, OpenAI, AsyncAzureOpenAI, AsyncOpenAI

from langchain_openai import AzureChatOpenAI, ChatOpenAI, AzureOpenAIEmbeddings, OpenAIEmbeddings

from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AsyncAzureOpenAI as sk_AsyncAzureOpenAI
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import AsyncOpenAI as sk_AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion

from config import settings

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
    embedding_name: str = "text-embedding-ada-002"
    org_id: Optional[str] = None

    def embeddings(self) -> Union[AzureOpenAIEmbeddings, OpenAIEmbeddings]:
        " Connect to the embeddings "
        if self.api_type == 'azure':
            embeddings = AzureOpenAIEmbeddings(
                api_key=self.api_key,
                api_version=self.api_version,
                organization=self.org_id,
                azure_endpoint=self.api_base,
                model=self.embedding_name)
        else:
            embeddings = OpenAIEmbeddings(
                api_key=self.api_key,
                organization=self.org_id,
                base_url=self.api_base,
                model=self.embedding_name)
        return embeddings

    def langchain(self) -> Union[ChatOpenAI, AzureChatOpenAI]:
        " Connect via Langchain "
        if self.api_type == 'azure':
            llm = AzureChatOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                organization=self.org_id,
                azure_endpoint=self.api_base,
                model=self.deployment_name)
        else:
            llm = ChatOpenAI(client=self.openai(), api_key=self.api_key, model=self.deployment_name) # Need to provide api_key here to stop validator from complaining that it's not set in env
        return llm

    def sk(self, service_id: str='default') -> Tuple[str, Union[AzureChatCompletion, OpenAIChatCompletion]]:
        " Connect via Semantic Kernel "
        # base url should be https://domain_name - that's it.
        # deployment gets added on as "/openai/deployments/{deployment}" to become base_url
        if self.api_type == "azure":
            client = sk_AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                organization=self.org_id,
                azure_endpoint=self.api_base,
                azure_deployment=self.deployment_name
            )
            service = AzureChatCompletion(
                api_key=self.api_key,
                api_version=self.api_version,
                service_id=service_id,
                endpoint=self.api_base,
                deployment_name=self.deployment_name,
                async_client=client
            )
        else:
            client = sk_AsyncOpenAI(api_key=self.api_key, organization=self.org_id, base_url=self.api_base)
            service = OpenAIChatCompletion(self.deployment_name, async_client=client, service_id=service_id)
        return service_id, service

    def openai(self, async_client: bool=False) -> Union[AzureOpenAI, OpenAI]:
        " Connect via the base OpenAI "
        if self.api_type == 'azure':
            if async_client:
                client = AsyncAzureOpenAI(
                    api_key=self.api_key,
                    api_version=self.api_version,
                    organization=self.org_id,
                    azure_endpoint=self.api_base
                )
            else:
                client = AzureOpenAI(
                    api_key=self.api_key,
                    api_version=self.api_version,
                    organization=self.org_id,
                    azure_endpoint=self.api_base
                )
        else:
            if async_client:
                client = AsyncOpenAI(
                    api_key=self.api_key,
                    organization=self.org_id,
                    base_url=self.api_base
                )
            else:
                client = OpenAI(
                    api_key=self.api_key,
                    organization=self.org_id,
                    base_url=self.api_base
                )
        return client

def llm_from_settings():
    llmConnector = LLMConnect(
        api_type=settings.openai_api_type,
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        deployment_name=settings.openai_deployment_name,
        org_id=settings.openai_org_id
    )
    return llmConnector