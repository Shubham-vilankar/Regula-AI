from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama inference
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_api_key: str = "ollama"
    ollama_model: str = "qwen3:14b"

    # Model behavior
    enable_thinking: bool = True
    max_output_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95

    # Langfuse (v4 SDK uses LANGFUSE_BASE_URL)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias="LANGFUSE_BASE_URL",
    )

    # App
    app_env: str = "dev"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
