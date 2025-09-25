"""Configuration management for BankingLLM system."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Database
    database_url: str = "sqlite:///./data/bank.db"
    db_pool_size: int = 5
    db_pool_timeout: int = 30

    # LLM Configuration
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    llm_timeout: int = 60
    llm_max_retries: int = 3

    # Excel Export
    excel_output_dir: str = "./data/exports"
    max_export_rows: int = 100000  # Limit for performance

    # Logging
    log_level: str = "INFO"

    # Data Generation
    num_clients: int = 10000
    num_accounts_per_client_range: tuple = (1, 3)
    num_transactions: int = 1000000

    # Performance Settings
    max_concurrent_requests: int = 10
    request_timeout: int = 300
    cache_ttl: int = 3600  # 1 hour

    # Security
    allowed_origins: list = ["*"]
    api_rate_limit: str = "100/minute"

    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()

# Ensure required directories exist
Path(settings.excel_output_dir).mkdir(parents=True, exist_ok=True)
Path("./data").mkdir(parents=True, exist_ok=True)