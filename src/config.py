"""Configuration management for BankingLLM system."""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Database
    database_url: str = "sqlite:///./data/bank.db"
    db_pool_size: int = 5
    db_pool_timeout: int = 30

    # LLM Configuration
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:14b"  # Upgraded to 14B for better SQL generation
    llm_timeout: int = 300  # Extended timeout for 14B model
    llm_max_retries: int = 2  # Reduced retries since local should be reliable

    # Excel Export
    excel_output_dir: str = "./data/exports"
    max_export_rows: int = 100000  # Limit for performance

    # Logging
    log_level: str = "INFO"

    # Data Generation
    num_clients: int = 10000
    num_accounts_per_client_range: tuple = (1, 3)
    num_transactions: int = 1000000

    # Business Rules and Validation
    min_client_age: int = 18
    max_client_age: int = 100
    max_daily_withdrawal_limit: float = 10000000.0  # 10M UZS
    max_single_transaction_amount: float = 50000000.0  # 50M UZS
    risk_score_threshold: float = 0.8  # Transactions above this are flagged
    kyc_verification_required_amount: float = 5000000.0  # 5M UZS

    # Account Business Rules
    checking_overdraft_limit: float = 1000000.0  # 1M UZS
    savings_minimum_balance: float = 50000.0  # 50K UZS
    business_minimum_balance: float = 500000.0  # 500K UZS

    # Compliance and AML
    suspicious_cash_threshold: float = 20000000.0  # 20M UZS
    multiple_transaction_threshold: int = 10  # Transactions per hour
    international_transfer_limit: float = 100000000.0  # 100M UZS

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