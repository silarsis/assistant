from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Literal

class AppSettings(BaseSettings):
    # Read the .env file
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
    
    # OpenAI/Azure API key and configuration
    openai_api_type: Literal["openai", "azure"] = "openai"
    openai_api_version: str = "2023-06-01-preview"
    openai_api_key: str = ""
    openai_api_base: Optional[str] = None
    openai_deployment_name: str = ""
    openai_org_id: Optional[str] = None
    # Image variables, in case they're different
    img_openai_api_type: Literal["openai", "azure"] = "openai"
    img_openai_api_version: str = "2023-06-01-preview"
    img_openai_api_key: str = ""
    img_openai_api_base: Optional[str] = None
    img_openai_org_id: Optional[str] = None
    
    # Memory type and engine
    memory: Literal["local", "motorhead"] = "local"
    # Motorhead for local chat memory
    motorhead_host: str = "localhost"
    motorhead_port: str = "8001"
    # ChromaDB for doc storage
    chroma_mode: Optional[Literal["local", "remote"]] = "local"
    chroma_host: Optional[str] = 'localhost'
    chroma_port: Optional[str] = '6000'
    
    # Voice engine to use for text to speech
    voice: Literal["None", "TTS", "OpenAI", "ElevenLabs"] = 'None'
    # Text to speech running in the local docker container
    tts_host: str = "localhost"
    tts_port: str = "5002"
    # Text to speech via ElevenLabs
    elevenlabs_api_key: str = ""
    elevenlabs_voice_1_id: str = ""
    elevenlabs_voice_2_id: str = ""
    
    # Other Miscellaneous API keys, perhaps we need a generic way to store these?
    # Wolfram Alpha API Key
    wolfram_alpha_appid: Optional[str] = None
    # Google Search API Key
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None
    # Apify for scraping websites
    apify_api_key: Optional[str] = None
    
    
settings = AppSettings()