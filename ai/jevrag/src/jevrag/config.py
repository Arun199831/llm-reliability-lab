"""Configuration for talking to Jev, Qdrant, and the LLM.

Every tunable lives here so a retry setting or a model swap is a .env edit,
not a code change.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Jev ---
    jev_api_key: SecretStr = Field(..., description="TypeSafe API key.")
    jev_base_url: str = "https://api.typesafe.ai/v1"
    jev_model: str = "jev-latest"
    jev_timeout_s: float = 10.0
    jev_connect_timeout_s: float = 3.0
    jev_max_attempts: int = 4
    jev_backoff_base_s: float = 0.5
    jev_backoff_max_s: float = 8.0

    # --- Retrieval: dense ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "jevrag_chunks"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # --- Retrieval: sparse + fusion ---
    bm25_index_path: str = "./data/bm25_index.pkl"
    top_k_dense: int = 10     # candidates pulled from Qdrant before fusion
    top_k_sparse: int = 10    # candidates pulled from BM25 before fusion
    rrf_k: int = 60           # damping constant -- see retrieval.py

    # --- Sufficiency gate ---
    sufficiency_threshold: float = 0.6

    # --- Generation ---
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 60.0
    llm_max_tokens: int = 600
    llm_temperature: float = 0.1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton -- import this, never instantiate Settings directly."""
    return Settings()
