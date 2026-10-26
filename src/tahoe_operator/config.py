from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_gateway_base_url: str = ""
    model_gateway_api_key: str = ""
    model_name: str = ""
    embedding_model: str = ""
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8001
    mcp_url: str = "http://127.0.0.1:8001/mcp"
    database_path: Path = PROJECT_ROOT / "data" / "tahoe.sqlite"
    corpus_path: Path = PROJECT_ROOT / "data" / "corpus"
    vector_store_path: Path = PROJECT_ROOT / "data" / "vector_store"
    fixture_path: Path = PROJECT_ROOT / "data" / "fixtures"
    allow_keyword_rag_fallback: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
