"""Configuration management for BankingLLM system."""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Database Configuration
    database_url: str = "sqlite:///./data/bank.db"  # Single database with all regional data
    db_pool_size: int = 5
    db_pool_timeout: int = 30
    database_dir: str = "./data"

    # Regional Configuration for Data Generation
    regions: dict = {
        "Tashkent": {"economic_weight": 0.32, "description": "Economic Capital - Financial Hub"},
        "Samarkand": {"economic_weight": 0.28, "description": "Tourist & Cultural Center"},
        "Bukhara": {"economic_weight": 0.18, "description": "Historic Trade Center"},
        "Andijan": {"economic_weight": 0.12, "description": "Agricultural Hub"},
        "Fergana": {"economic_weight": 0.08, "description": "Manufacturing Center"},
        "Namangan": {"economic_weight": 0.06, "description": "Regional Center"},
        "Nukus": {"economic_weight": 0.04, "description": "Remote Region - Karakalpakstan"}
    }

    # Regional Economic and Demographic Modeling
    regional_demographics: dict = {
        "tashkent": {
            "population_percent": 0.35,  # 35% of clients
            "gdp_per_capita_multiplier": 2.5,  # Higher wealth
            "digital_adoption": 0.8,     # 80% digital banking usage
            "primary_industries": ["Finance", "IT", "Government", "Trade"],
            "occupation_weights": {
                "Bank Executive": 0.15, "IT Executive": 0.12, "Business Owner": 0.20,
                "Government Official": 0.10, "Doctor/Surgeon": 0.08, "Engineer": 0.10,
                "Import/Export Trader": 0.08, "Real Estate Agent": 0.07, "Student": 0.10
            }
        },
        "samarkand": {
            "population_percent": 0.28,
            "gdp_per_capita_multiplier": 1.8,
            "digital_adoption": 0.65,
            "primary_industries": ["Tourism", "Healthcare", "Education", "Government"],
            "occupation_weights": {
                "Doctor/Surgeon": 0.18, "Government Official": 0.15, "Teacher/Professor": 0.12,
                "Business Owner": 0.15, "Retail Manager": 0.10, "Engineer": 0.08,
                "Restaurant Owner": 0.12, "Student": 0.10
            }
        },
        "bukhara": {
            "population_percent": 0.18,
            "gdp_per_capita_multiplier": 1.6,
            "digital_adoption": 0.55,
            "primary_industries": ["Trade", "Tourism", "Textiles", "Agriculture"],
            "occupation_weights": {
                "Import/Export Trader": 0.20, "Business Owner": 0.18, "Restaurant Owner": 0.12,
                "Retail Manager": 0.12, "Agriculture Specialist": 0.10, "Construction Contractor": 0.08,
                "Teacher/Professor": 0.08, "Transportation Business": 0.12
            }
        },
        "andijan": {
            "population_percent": 0.12,
            "gdp_per_capita_multiplier": 1.2,
            "digital_adoption": 0.45,
            "primary_industries": ["Agriculture", "Construction", "Manufacturing"],
            "occupation_weights": {
                "Agriculture Specialist": 0.25, "Construction Contractor": 0.18, "Engineer": 0.12,
                "Retail Manager": 0.10, "Transportation Business": 0.15, "Teacher/Professor": 0.10,
                "Government Official": 0.10
            }
        },
        "fergana": {
            "population_percent": 0.08,
            "gdp_per_capita_multiplier": 1.3,
            "digital_adoption": 0.50,
            "primary_industries": ["Manufacturing", "Agriculture", "Energy"],
            "occupation_weights": {
                "Engineer": 0.20, "Agriculture Specialist": 0.15, "Construction Contractor": 0.15,
                "Retail Manager": 0.12, "Transportation Business": 0.12, "Teacher/Professor": 0.10,
                "Government Official": 0.08, "Business Owner": 0.08
            }
        },
        "namangan": {
            "population_percent": 0.06,
            "gdp_per_capita_multiplier": 1.1,
            "digital_adoption": 0.40,
            "primary_industries": ["Agriculture", "Education", "Government"],
            "occupation_weights": {
                "Agriculture Specialist": 0.20, "Teacher/Professor": 0.18, "Government Official": 0.15,
                "Retail Manager": 0.12, "Construction Contractor": 0.10, "Engineer": 0.08,
                "Transportation Business": 0.10, "Retired": 0.07
            }
        },
        "nukus": {
            "population_percent": 0.04,
            "gdp_per_capita_multiplier": 0.9,
            "digital_adoption": 0.30,
            "primary_industries": ["Government", "Agriculture", "Education"],
            "occupation_weights": {
                "Government Official": 0.25, "Agriculture Specialist": 0.20, "Teacher/Professor": 0.15,
                "Retired": 0.12, "Construction Contractor": 0.08, "Retail Manager": 0.08,
                "Transportation Business": 0.12
            }
        }
    }

    # Cultural and Seasonal Banking Patterns
    seasonal_patterns: dict = {
        "samarkand": {
            "tourism_months": [4, 5, 6, 7, 8, 9],  # Spring through early fall
            "transaction_multiplier": 1.4  # 40% increase during tourism season
        },
        "andijan": {
            "harvest_months": [8, 9, 10],  # Harvest season
            "transaction_multiplier": 1.8  # 80% increase during harvest
        },
        "fergana": {
            "industrial_cycles": [1, 4, 7, 10],  # Quarterly industrial cycles
            "transaction_multiplier": 1.2
        }
    }

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
    num_transactions: int = 1200000

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

    # Enhanced Transaction Realism
    merchant_categories: dict = {
        "5411": {"name": "Grocery Stores", "weight": 0.15, "avg_amount": 150000},
        "5541": {"name": "Service Stations", "weight": 0.12, "avg_amount": 250000},
        "5912": {"name": "Drug Stores", "weight": 0.08, "avg_amount": 75000},
        "5999": {"name": "Miscellaneous Retail", "weight": 0.10, "avg_amount": 300000},
        "6011": {"name": "ATM Cash Withdrawal", "weight": 0.20, "avg_amount": 500000},
        "4814": {"name": "Telecom Services", "weight": 0.05, "avg_amount": 100000},
        "5814": {"name": "Fast Food", "weight": 0.08, "avg_amount": 80000},
        "5661": {"name": "Shoe Stores", "weight": 0.03, "avg_amount": 400000},
        "5732": {"name": "Electronics", "weight": 0.04, "avg_amount": 800000},
        "5311": {"name": "Department Stores", "weight": 0.06, "avg_amount": 500000},
        "5921": {"name": "Package Stores", "weight": 0.02, "avg_amount": 120000},
        "5812": {"name": "Restaurants", "weight": 0.07, "avg_amount": 200000}
    }

    transaction_velocity_patterns: dict = {
        "MORNING_RUSH": {"hours": [7, 8, 9], "multiplier": 1.8},
        "LUNCH_BREAK": {"hours": [12, 13], "multiplier": 1.5},
        "EVENING_PEAK": {"hours": [17, 18, 19], "multiplier": 1.6},
        "NIGHT_LOW": {"hours": [22, 23, 0, 1, 2, 3, 4, 5], "multiplier": 0.3}
    }

    seasonal_banking_cycles: dict = {
        "SALARY_WEEKS": {"days": [1, 2, 15, 16], "deposit_multiplier": 2.5},
        "MONTH_END": {"days": [28, 29, 30, 31], "withdrawal_multiplier": 1.8},
        "HOLIDAY_SEASON": {"months": [11, 12, 1], "transaction_multiplier": 1.4},
        "RAMADAN_PATTERNS": {"month": 4, "evening_multiplier": 2.0}
    }

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