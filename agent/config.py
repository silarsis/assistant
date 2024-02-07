from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource
from pydantic.fields import FieldInfo
from pathlib import Path

from typing import Optional, Literal, Tuple, Any, Dict, Type

import json

class JsonConfigSettingsSource(PydanticBaseSettingsSource): # Taken from https://docs.pydantic.dev/latest/concepts/pydantic_settings/#adding-sources
    """
    A simple settings source class that loads variables from a JSON file
    at the project's root.

    Here we happen to choose to use the `env_file_encoding` from Config
    when reading `config.json`
    """

    def get_field_value(self, field: FieldInfo, field_name: str) -> Tuple[Any, str, bool]:
        encoding = self.config.get('env_file_encoding')
        try:
            file_content_json = json.loads(Path('.env.json').read_text(encoding))
        except FileNotFoundError:
            pass # We can ignore if it's not there.
        field_value = file_content_json.get(field_name)
        return field_value, field_name, False

    def prepare_field_value(self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool) -> Any:
        return value

    def __call__(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        try:
            for field_name, field in self.settings_cls.model_fields.items():
                field_value, field_key, value_is_complex = self.get_field_value(
                    field, field_name
                )
                field_value = self.prepare_field_value(
                    field_name, field, field_value, value_is_complex
                )
                if field_value is not None:
                    d[field_key] = field_value
        except json.decoder.JSONDecodeError as e:
            print(f"Failed to load JSON, skipping: {e}")
        return d

class AppSettings(BaseSettings):
    # Read the .env file
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8', 
        extra='ignore', 
        revalidate_instances='always',
        validate_default=True
    )
    # XXX Can I load the contents of the character file here as a variable?
    
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
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            dotenv_settings,
            env_settings,
            JsonConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    def save(self):
        with open('.env.json', 'w') as f:
            f.write(json.dumps(self.model_dump(mode='json')))
    
settings = AppSettings()