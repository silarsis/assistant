from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource
from pydantic_settings.sources import EnvSettingsSource
from pydantic.fields import FieldInfo, Field
from pydantic import computed_field, BaseModel
from pathlib import Path
import keyring
from keyring.backends.SecretService import Keyring as SecretServiceKeyring

from typing import Optional, Literal, Tuple, Any, Dict, Type, List, Mapping, Iterator, cast

import json

KEYRING_NAME = "EchoAgentKeys"
SECRET_KEYS = ["openai_api_key", "img_openai_api_key", "img_upload_api_key", "elevenlabs_api_key",
             "wolfram_alpha_appid", "google_api_key", "google_cse_id", "apify_api_key", "presto_password"]
EXCLUDE_KEYS = SECRET_KEYS + ['spotify_client_verifier', 'spotify_client_code', 'spotify_access_token', 'spotify_refresh_token', 'spotify_expiry']

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
            file_content_json = {} # We can ignore if it's not there.
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
    
class KeyringSettingsSource(EnvSettingsSource):  # From https://github.com/pydantic/pydantic-settings/pull/140/files
    """
    Source class for loading settings values from keyrings.
    """
    
    def __init__(
        self,
        settings_cls: type[BaseSettings],
        keyring_backend: str | None = None,
        case_sensitive: bool | None = None,
        env_prefix: str | None = None,
        env_nested_delimiter: str | None = None,
    ) -> None:
        self.keyring_backend = (
            keyring_backend if keyring_backend is not None else settings_cls.model_config.get('keyring_backend')
        )
        super().__init__(settings_cls, case_sensitive, env_prefix, env_nested_delimiter)

    def _load_env_vars(self) -> Mapping[str, str | None]:
        return self._read_keyring(self.case_sensitive)

    def _read_keyring(self, case_sensitive: bool) -> Mapping[str, str | None]:
        keyring_vars: dict[str, str | None] = {}
        kr = cast(SecretServiceKeyring, keyring.core.get_keyring())
        for fieldname in SECRET_KEYS:
            keyring_vars[fieldname] = kr.get_password(KEYRING_NAME, fieldname)
        if not case_sensitive:
            return {k.lower(): v for k, v in keyring_vars.items()}
        else:
            return keyring_vars

    def __call__(self) -> dict[str, Any]:
        data: dict[str, Any] = super().__call__()

        data_lower_keys: list[str] = []
        if not self.case_sensitive:
            data_lower_keys = [x.lower() for x in data.keys()]

        # As `extra` config is allowed in keyring settings source, We have to
        # update data with extra env variables from keyring.
        for env_name, env_value in self.env_vars.items():
            if env_name.startswith(self.env_prefix) and env_value is not None:
                env_name_without_prefix = env_name[self.env_prefix_len :]
                first_key, *_ = env_name_without_prefix.split(self.env_nested_delimiter)

                if (data_lower_keys and first_key not in data_lower_keys) or (
                    not data_lower_keys and first_key not in data
                ):
                    data[first_key] = env_value

        return data

    def __repr__(self) -> str:
        return (
            f'KeyringSettingsSource(keyring_backend={self.keyring_backend!r}, '
            f'env_nested_delimiter={self.env_nested_delimiter!r}, env_prefix_len={self.env_prefix_len!r})'
        )

class AgentModel(BaseModel):
    role: str
    goal: str
    backstory: str

class AppSettings(BaseSettings):
    # Read the .env file
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8', 
        extra='ignore', 
        revalidate_instances='always',
        validate_default=True
    )
    character: str = """You are an AI assistant.
Your name is Echo.
You are designed to be helpful, but you are also a bit of a smartass. You don't have to be polite.
Your goal is to provide the user with answers to their questions and sometimes make them laugh.
You have a variety of tools, plus a planner, plus a crew of other AI that can help you answer questions.
Use Markdown formatting in your answers where appropriate.
You should answer in a personalised way, not too formal, concise, and not always polite.
Always check the context and chat history first to see if you know an answer.
"""
    # CrewAI Agents
    crew: List[AgentModel] = []
    
    # OpenAI/Azure API key and configuration
    openai_api_type: Literal["openai", "azure"] = "openai"
    openai_api_version: str = "2023-06-01-preview"
    openai_api_key: str = Field(default="NOT_PROVIDED", min_length=3)
    openai_api_base: Optional[str] = None
    openai_deployment_name: str = "gpt-4"
    openai_org_id: Optional[str] = None
    # Image generation variables, in case they're different
    img_openai_inherit: bool = True # Inherit from the main OpenAI settings
    img_openai_api_type: Literal["openai", "azure"] = "openai"
    img_openai_api_version: str = "2023-06-01-preview"
    img_openai_api_key: str = ""
    img_openai_api_base: Optional[str] = None
    img_openai_deployment_name: str = ""
    img_openai_org_id: Optional[str] = None
    # Image upload api key for talking direct to OpenAI
    img_upload_api_key: str = ""
    
    # ChromaDB for doc storage
    chroma_mode: Optional[Literal["local", "remote"]] = "local"
    chroma_host: Optional[str] = 'localhost'
    chroma_port: Optional[str] = '6000'
    
    # Voice engine to use for text to speech
    voice: Literal["None", "ElevenLabs", "OpenAI", "TTS"] = 'None'
    # Text to speech running in the local docker container
    tts_host: str = "localhost"
    tts_port: str = "5002"
    # Text to speech via ElevenLabs
    elevenlabs_api_key: str = ""
    elevenlabs_voice_1_id: str = ""
    elevenlabs_voice_2_id: str = ""
    
    # Client settings
    hear_thoughts: bool = False
    
    # Other Miscellaneous API keys, perhaps we need a generic way to store these?
    # Wolfram Alpha API Key
    wolfram_alpha_appid: Optional[str] = None
    # Google Search API Key
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None
    # Apify for scraping websites
    apify_api_key: Optional[str] = None
    # PyHive / Presto for internal data access
    presto_host: str = ""
    presto_username: str = ""
    # Spotify for radio
    spotify_client_id: str = "afffe002f50944d792eb41d79a5b5a96"
    spotify_client_verifier: str = ''
    spotify_client_code: str = ''
    spotify_access_token: str = ''
    spotify_refresh_token: str = ''
    spotify_expiry: int = 0
    
    @computed_field(repr=False)
    def presto_password(self) -> str:
        return keyring.get_password(KEYRING_NAME, "presto_password") or ""
    
    @presto_password.setter
    def presto_password(self, value: str) -> None:
        keyring.set_password(KEYRING_NAME, "presto_password", value)
    
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
            KeyringSettingsSource(settings_cls, keyring_backend=KEYRING_NAME),
            file_secret_settings,
        )

    def save(self):
        print("Saving Settings")
        with open('.env.json', 'w') as f:
            f.write(json.dumps(self.model_dump(mode='json', exclude=EXCLUDE_KEYS)))
        for fieldname, field in self.model_fields.items():
            if fieldname in SECRET_KEYS:
                keyring.set_password(KEYRING_NAME, fieldname, getattr(self, fieldname))
    
settings = AppSettings()