"""Professional Single Database Layer for Banking Intelligence System.

This module provides comprehensive database models and operations specifically
designed for banking and financial data analysis with all regional data
consolidated into a single optimized database.

Features:
- SQLAlchemy ORM models for banking entities
- Single database with regional indexing and optimization
- Professional data generation with realistic banking patterns
- Comprehensive error handling and validation
- Connection pooling and performance optimization
- Enhanced LLM context awareness with real data sampling
"""

import random
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
from contextlib import contextmanager

import numpy as np
from faker import Faker
from loguru import logger
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    BigInteger,
    String,
    Numeric,
    create_engine,
    func,
    text,
    inspect,
)
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError

from .config import settings
from .exceptions import (
    DatabaseError,
    DatabaseConnectionError,
    QueryExecutionError,
    ValidationError
)

Base = declarative_base()

# Professional Currency Conversion Utilities for Uzbekistan Banking
# Official exchange: 1 сўм (UZS) = 100 тийин

def sum_to_tiyin(sum_amount: Union[float, int, Decimal]) -> int:
    """Convert сўм amount to тийин for precise financial calculations.

    Args:
        sum_amount: Amount in сўм (UZS major currency unit)

    Returns:
        Amount in тийин (UZS minor currency unit)

    Raises:
        ValidationError: If amount is invalid
    """
    if sum_amount is None:
        raise ValidationError("Currency amount cannot be None", field_name="sum_amount")

    try:
        return int(round(float(sum_amount) * 100))
    except (ValueError, TypeError) as e:
        raise ValidationError(
            f"Invalid currency amount: {sum_amount}",
            field_name="sum_amount",
            invalid_value=str(sum_amount)
        )

def tiyin_to_sum(tiyin_amount: Union[int, float]) -> float:
    """Convert тийин amount to сўм for display purposes.

    Args:
        tiyin_amount: Amount in тийин (UZS minor currency unit)

    Returns:
        Amount in сўм (UZS major currency unit)

    Raises:
        ValidationError: If amount is invalid
    """
    if tiyin_amount is None:
        raise ValidationError("Currency amount cannot be None", field_name="tiyin_amount")

    try:
        return float(tiyin_amount) / 100.0
    except (ValueError, TypeError) as e:
        raise ValidationError(
            f"Invalid tiyin amount: {tiyin_amount}",
            field_name="tiyin_amount",
            invalid_value=str(tiyin_amount)
        )

def format_currency(
    tiyin_amount: Optional[int],
    language: str = "english",
    show_tiyin: bool = False
) -> str:
    """Format tiyin amount as localized currency string.

    Args:
        tiyin_amount: Amount in тийин to format
        language: Display language ('english', 'russian', 'uzbek')
        show_tiyin: Whether to show minor currency units

    Returns:
        Formatted currency string with appropriate symbols and formatting
    """
    if tiyin_amount is None:
        return "0 UZS" if language == "english" else ("0 сум" if language == "russian" else "0 so'm")

    sum_amount = tiyin_to_sum(tiyin_amount)

    # Language-specific currency symbols
    if language.lower() == "russian":
        currency_symbol = "сум"
        minor_currency = "тийин"
    elif language.lower() == "uzbek":
        currency_symbol = "so'm"
        minor_currency = "tiyin"
    else:  # Default to English
        currency_symbol = "UZS"
        minor_currency = "tiyin"

    if show_tiyin and tiyin_amount % 100 != 0:
        # Show both major and minor currency units
        sum_part = int(sum_amount)
        tiyin_part = tiyin_amount % 100
        return f"{sum_part:,} {currency_symbol} {tiyin_part} {minor_currency}"
    else:
        # Show in decimal format: 2000.30 UZS
        return f"{sum_amount:,.2f} {currency_symbol}"

def generate_uzbek_phone_number(phone_type="mobile", region=None):
    """Generate realistic Uzbek phone numbers with proper formatting."""
    if phone_type == "mobile":
        # Major Uzbek mobile operators: 90, 91, 93, 94, 95, 97, 98, 99
        operator = random.choice([90, 91, 93, 94, 95, 97, 98, 99])
        number = f"{random.randint(1000000, 9999999):07d}"
        return f"+998{operator}{number}"
    else:  # landline
        # Area codes for major Uzbek cities
        area_codes = {
            "Tashkent": 71,
            "Samarkand": 62,
            "Bukhara": 65,
            "Fergana": 69,
            "Andijan": 74,
            "Namangan": 69,
            "Nukus": 61,
            "Qarshi": 75,
            "Urgench": 62
        }

        # Use region-specific area code if provided, otherwise random
        if region and region in area_codes:
            area_code = area_codes[region]
        else:
            area_code = random.choice(list(area_codes.values()))

        number = f"{random.randint(1000000, 9999999):07d}"
        return f"+998{area_code}{number}"

# Fake instance for location generation
fake = Faker()


class Branch(Base):
    """Simplified branch model with essential banking fields only."""

    __tablename__ = "branches"

    id = Column(Integer, primary_key=True, index=True)
    branch_code = Column(String(10), unique=True, nullable=False, index=True)
    branch_name = Column(String(100), nullable=False)
    city = Column(String(50), nullable=False, index=True)
    region = Column(String(50), nullable=False, index=True)
    branch_type = Column(String(20), nullable=False, default="FULL_SERVICE", index=True)
    status = Column(String(20), nullable=False, default="ACTIVE", index=True)
    daily_cash_limit = Column(BigInteger, nullable=True)
    max_transaction_amount = Column(BigInteger, nullable=True)
    opened_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_by = Column(String(50), nullable=False, default="SYSTEM")

    # Relationships
    accounts = relationship("Account", back_populates="branch")

    def __repr__(self):
        return f"<Branch(id={self.id}, code='{self.branch_code}', name='{self.branch_name}', city='{self.city}')>"


class Product(Base):
    """Simplified product model with essential banking fields only."""

    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    product_code = Column(String(20), unique=True, nullable=False, index=True)
    product_name = Column(String(100), nullable=False)
    product_category = Column(String(30), nullable=False, index=True)
    product_type = Column(String(30), nullable=False, index=True)
    description = Column(String(500), nullable=True)
    base_interest_rate = Column(Numeric(5, 4), nullable=True, default=0.0)
    annual_fee = Column(BigInteger, nullable=True, default=0)
    minimum_balance = Column(BigInteger, nullable=True, default=0)
    risk_category = Column(String(20), nullable=False, default="MEDIUM", index=True)
    status = Column(String(20), nullable=False, default="ACTIVE", index=True)
    launch_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_by = Column(String(50), nullable=False, default="SYSTEM")

    def __repr__(self):
        return f"<Product(id={self.id}, code='{self.product_code}', name='{self.product_name}', category='{self.product_category}')>"


class Client(Base):
    """Simplified client model with essential banking fields only."""

    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    client_number = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    birth_date = Column(DateTime, nullable=False)
    email = Column(String(100), nullable=True, index=True)
    region = Column(String(50), nullable=False, index=True)
    occupation = Column(String(100), nullable=True)
    income_level = Column(String(20), nullable=True, index=True)
    risk_rating = Column(String(10), nullable=False, default="MEDIUM", index=True)
    status = Column(String(20), nullable=False, default="ACTIVE", index=True)
    kyc_status = Column(String(20), nullable=False, default="PENDING")
    created_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    created_by = Column(String(50), nullable=False, default="SYSTEM")

    # Relationships
    accounts = relationship("Account", back_populates="client", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Client(id={self.id}, client_number='{self.client_number}', name='{self.name}', region='{self.region}')>"


class Account(Base):
    """Simplified account model with essential banking fields only."""

    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    account_number = Column(String(20), unique=True, nullable=False, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False, index=True)
    branch_id = Column(Integer, ForeignKey("branches.id"), nullable=False, index=True)
    account_type = Column(String(20), nullable=False, index=True)
    account_subtype = Column(String(30), nullable=True)
    balance = Column(BigInteger, nullable=False, default=0, index=True)
    available_balance = Column(BigInteger, nullable=False, default=0)
    interest_rate = Column(Numeric(5, 4), nullable=True, default=0.0)
    minimum_balance = Column(BigInteger, nullable=True, default=0)
    status = Column(String(20), nullable=False, default="ACTIVE", index=True)
    open_date = Column(DateTime, nullable=False, index=True)
    created_date = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    client = relationship("Client", back_populates="accounts")
    branch = relationship("Branch", back_populates="accounts")
    transactions = relationship("Transaction", back_populates="account", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Account(id={self.id}, account_number='{self.account_number}', type='{self.account_type}', balance={self.balance})>"


class Transaction(Base):
    """Simplified transaction model with essential banking fields only."""

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_reference = Column(String(30), unique=True, nullable=False, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False, index=True)
    transaction_type = Column(String(30), nullable=False, index=True)
    transaction_subtype = Column(String(50), nullable=True)
    amount = Column(BigInteger, nullable=False, index=True)
    fee_amount = Column(BigInteger, nullable=False, default=0)
    channel = Column(String(20), nullable=False, index=True)
    balance_before = Column(BigInteger, nullable=False)
    balance_after = Column(BigInteger, nullable=False)
    risk_score = Column(Numeric(3, 3), nullable=True)
    flagged_for_review = Column(Boolean, default=False, index=True)
    review_status = Column(String(20), nullable=True)
    transaction_date = Column(DateTime, nullable=False, index=True)
    processing_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    status = Column(String(20), nullable=False, default="COMPLETED", index=True)
    created_by = Column(String(50), nullable=False, default="SYSTEM")
    authorized_by = Column(String(50), nullable=True)

    # Relationships
    account = relationship("Account", back_populates="transactions")

    def __repr__(self):
        return f"<Transaction(id={self.id}, ref='{self.transaction_reference}', type='{self.transaction_type}', amount={self.amount})>"


class DatabaseManager:
    """Professional single database manager with comprehensive error handling and optimization.

    This class manages database connections, query execution, and data generation
    for the banking intelligence system using a single optimized database.

    Attributes:
        database_url: Database connection URL
        engine: SQLAlchemy engine instance
        SessionLocal: Session factory for database operations
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        """Initialize database manager with optimized configuration.

        Args:
            database_url: Optional database URL override

        Raises:
            DatabaseConnectionError: If database connection fails
            ValidationError: If configuration is invalid
        """
        self.database_url = database_url or settings.database_url

        if not self.database_url:
            raise ValidationError(
                "Database URL is required",
                field_name="database_url"
            )

        logger.info(f"Initializing database manager: {self._mask_url(self.database_url)}")

        try:
            # Optimized engine configuration with SQLite performance pragmas
            connect_args = {}
            if "sqlite" in self.database_url:
                connect_args = {
                    "check_same_thread": False,
                    "timeout": 30.0,
                }

            self.engine = create_engine(
                self.database_url,
                pool_size=getattr(settings, 'db_pool_size', 5),
                max_overflow=10,
                pool_timeout=getattr(settings, 'db_pool_timeout', 30),
                pool_recycle=3600,  # Recycle connections after 1 hour
                echo=getattr(settings, 'log_level', 'INFO') == "DEBUG",
                connect_args=connect_args
            )

            # Test database connection
            self._test_connection()

            # Apply SQLite performance pragmas
            if "sqlite" in self.database_url:
                self._optimize_sqlite()

            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
                expire_on_commit=False  # Performance optimization
            )

            logger.info("✅ Database manager initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize database manager: {str(e)}")
            raise DatabaseConnectionError(
                f"Failed to initialize database connection: {str(e)}",
                database_name=self._mask_url(self.database_url)
            )

    def _mask_url(self, url: str) -> str:
        """Mask sensitive information in database URL for logging.

        Args:
            url: Database URL to mask

        Returns:
            Masked URL safe for logging
        """
        if not url:
            return "None"

        # Mask password in connection strings
        if "://" in url and "@" in url:
            parts = url.split("://")
            if len(parts) == 2:
                protocol = parts[0]
                rest = parts[1]
                if "@" in rest:
                    auth_part = rest.split("@")[0]
                    if ":" in auth_part:
                        user = auth_part.split(":")[0]
                        return f"{protocol}://{user}:***@{rest.split('@')[1]}"
        return url

    def _test_connection(self) -> None:
        """Test database connection and raise appropriate errors.

        Raises:
            DatabaseConnectionError: If connection test fails
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.debug("Database connection test successful")
        except Exception as e:
            raise DatabaseConnectionError(
                f"Database connection test failed: {str(e)}",
                database_name=self._mask_url(self.database_url)
            )

    def _optimize_sqlite(self):
        """Apply SQLite-specific performance optimizations."""
        with self.engine.connect() as conn:
            # Use WAL mode for better concurrent access
            conn.execute(text("PRAGMA journal_mode=WAL"))
            # Faster synchronization for better performance
            conn.execute(text("PRAGMA synchronous=NORMAL"))
            # Larger cache size for better performance (1GB)
            conn.execute(text("PRAGMA cache_size=1000000"))
            # Use memory for temporary storage
            conn.execute(text("PRAGMA temp_store=memory"))
            # Optimize for bulk inserts
            conn.execute(text("PRAGMA optimize"))
            conn.commit()
            logger.info("Applied SQLite performance optimizations")

    def create_tables(self, drop_existing=False):
        """Create all database tables."""
        logger.info("Creating database tables...")
        if drop_existing:
            logger.info("Dropping existing tables first...")
            Base.metadata.drop_all(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        self._create_performance_indexes()
        logger.info("Database tables created successfully")

    def _create_performance_indexes(self):
        """Create advanced indexes for optimal query performance."""
        logger.info("Creating advanced performance indexes...")

        session = self.get_session()
        try:
            # Banking Analytics Composite Indexes for Fast Queries
            performance_indexes = [
                # Client analysis indexes
                "CREATE INDEX IF NOT EXISTS idx_client_region_income_risk ON clients(region, income_level, risk_rating)",
                "CREATE INDEX IF NOT EXISTS idx_client_occupation_income ON clients(occupation, income_level)",
                "CREATE INDEX IF NOT EXISTS idx_client_status_kyc ON clients(status, kyc_status)",
                "CREATE INDEX IF NOT EXISTS idx_client_created_region ON clients(created_date, region)",

                # Account analysis indexes
                "CREATE INDEX IF NOT EXISTS idx_account_type_balance ON accounts(account_type, balance)",
                "CREATE INDEX IF NOT EXISTS idx_account_client_status ON accounts(client_id, status)",
                "CREATE INDEX IF NOT EXISTS idx_account_branch_type_balance ON accounts(branch_id, account_type, balance)",
                "CREATE INDEX IF NOT EXISTS idx_account_balance_status ON accounts(balance DESC, status)",
                "CREATE INDEX IF NOT EXISTS idx_account_open_date_type ON accounts(open_date, account_type)",

                # Transaction performance indexes with sequence numbers
                "CREATE INDEX IF NOT EXISTS idx_txn_account_sequence ON transactions(account_id, sequence_number DESC)",
                "CREATE INDEX IF NOT EXISTS idx_txn_date_type_amount ON transactions(transaction_date, transaction_type, amount)",
                "CREATE INDEX IF NOT EXISTS idx_txn_account_date_sequence ON transactions(account_id, transaction_date DESC, sequence_number DESC)",
                "CREATE INDEX IF NOT EXISTS idx_txn_amount_type_channel ON transactions(amount, transaction_type, channel)",
                "CREATE INDEX IF NOT EXISTS idx_txn_risk_flagged_amount ON transactions(risk_score DESC, flagged_for_review, amount)",
                "CREATE INDEX IF NOT EXISTS idx_txn_subtype_amount_date ON transactions(transaction_subtype, amount, transaction_date)",
                "CREATE INDEX IF NOT EXISTS idx_txn_channel_type_date ON transactions(channel, transaction_type, transaction_date)",
                "CREATE INDEX IF NOT EXISTS idx_txn_flagged_risk_amount ON transactions(flagged_for_review, risk_score DESC, amount DESC)",

                # Complex analytics indexes for professional queries (removed non-deterministic strftime function)
                # "CREATE INDEX IF NOT EXISTS idx_txn_date_month_type ON transactions(strftime('%Y-%m', transaction_date), transaction_type)",
                "CREATE INDEX IF NOT EXISTS idx_txn_large_amounts ON transactions(amount) WHERE ABS(amount) > 5000000",  # 50K сўм in tiyin
                "CREATE INDEX IF NOT EXISTS idx_txn_cash_deposits ON transactions(transaction_subtype, amount) WHERE transaction_subtype = 'CASH_DEPOSIT'",

                # Multi-table join optimization indexes
                "CREATE INDEX IF NOT EXISTS idx_branch_region_type ON branches(region, branch_type)",
                "CREATE INDEX IF NOT EXISTS idx_product_category_type ON products(product_category, product_type)",

                # Risk and compliance specific indexes
                "CREATE INDEX IF NOT EXISTS idx_txn_risk_compliance ON transactions(risk_score, flagged_for_review, transaction_date)",
                "CREATE INDEX IF NOT EXISTS idx_client_high_risk ON clients(risk_rating, income_level) WHERE risk_rating = 'HIGH'",

                # Time-based performance indexes (removed non-deterministic date functions)
                # "CREATE INDEX IF NOT EXISTS idx_txn_recent_90days ON transactions(transaction_date, account_id) WHERE transaction_date >= date('now', '-90 days')",
                # "CREATE INDEX IF NOT EXISTS idx_txn_recent_30days ON transactions(transaction_date, transaction_type) WHERE transaction_date >= date('now', '-30 days')",

                # Specific business scenario indexes for impressive demos
                "CREATE INDEX IF NOT EXISTS idx_client_occupation_risk_income ON clients(occupation, risk_rating, income_level)",
                "CREATE INDEX IF NOT EXISTS idx_txn_structuring_detection ON transactions(transaction_type, amount, transaction_date) WHERE transaction_type = 'DEPOSIT' AND amount BETWEEN 4500000 AND 4999999"  # 45K-50K сўм in tiyin
            ]

            for index_sql in performance_indexes:
                try:
                    session.execute(text(index_sql))
                    session.commit()
                except Exception as e:
                    logger.warning(f"Index creation warning (may already exist): {e}")
                    session.rollback()

            # Analyze tables for query planner optimization
            analyze_commands = [
                "ANALYZE clients",
                "ANALYZE accounts",
                "ANALYZE transactions",
                "ANALYZE branches",
                "ANALYZE products"
            ]

            for analyze_sql in analyze_commands:
                session.execute(text(analyze_sql))
                session.commit()

            logger.info("Advanced performance indexes created successfully")

        except Exception as e:
            logger.error(f"Error creating performance indexes: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def _database_has_data(self) -> bool:
        """Check if database already contains data."""
        session = self.get_session()
        try:
            # Check if tables exist and have data
            try:
                client_count = session.query(Client).count()
                transaction_count = session.query(Transaction).count()
                return client_count > 0 or transaction_count > 0
            except Exception:
                # Tables don't exist yet
                return False
        finally:
            session.close()

    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()

    def _pre_compute_client_profiles(self, clients: List[Client]) -> dict:
        """Pre-compute all client profiles for ultra-fast transaction generation."""
        logger.info("Pre-computing client profiles for high-performance generation...")

        # Define occupation profiles (moved from generate_mock_data for reuse)
        occupation_profiles = {
            "IT Executive": {"income_weights": [0.05, 0.20, 0.50, 0.25], "avg_accounts": 2.5, "tech_savvy": True, "regions": ["Tashkent", "Samarkand"]},
            "Bank Executive": {"income_weights": [0.02, 0.15, 0.60, 0.23], "avg_accounts": 3.2, "tech_savvy": True, "regions": ["Tashkent", "Samarkand", "Bukhara"]},
            "Doctor/Surgeon": {"income_weights": [0.03, 0.25, 0.55, 0.17], "avg_accounts": 2.8, "tech_savvy": False, "regions": ["Tashkent", "Samarkand", "Andijan"]},
            "Business Owner": {"income_weights": [0.10, 0.25, 0.45, 0.20], "avg_accounts": 3.8, "tech_savvy": True, "regions": ["Tashkent", "Samarkand", "Bukhara"]},
            "Government Official": {"income_weights": [0.15, 0.45, 0.35, 0.05], "avg_accounts": 2.2, "tech_savvy": False, "regions": ["Tashkent", "Namangan", "Fergana"]},
            "Engineer": {"income_weights": [0.20, 0.50, 0.25, 0.05], "avg_accounts": 2.1, "tech_savvy": True, "regions": ["Tashkent", "Samarkand", "Andijan"]},
            "Teacher/Professor": {"income_weights": [0.40, 0.45, 0.13, 0.02], "avg_accounts": 1.8, "tech_savvy": False, "regions": ["Tashkent", "Samarkand", "Bukhara", "Namangan"]},
            "Retail Manager": {"income_weights": [0.25, 0.55, 0.18, 0.02], "avg_accounts": 2.0, "tech_savvy": True, "regions": ["Tashkent", "Samarkand", "Andijan", "Fergana"]},
            "Construction Contractor": {"income_weights": [0.30, 0.45, 0.20, 0.05], "avg_accounts": 2.3, "tech_savvy": False, "regions": ["Tashkent", "Bukhara", "Nukus"]},
            "Agriculture Specialist": {"income_weights": [0.45, 0.40, 0.13, 0.02], "avg_accounts": 1.9, "tech_savvy": False, "regions": ["Fergana", "Andijan", "Namangan", "Nukus"]},
            "Import/Export Trader": {"income_weights": [0.15, 0.30, 0.40, 0.15], "avg_accounts": 3.5, "tech_savvy": True, "regions": ["Tashkent", "Samarkand"]},
            "Restaurant Owner": {"income_weights": [0.25, 0.45, 0.25, 0.05], "avg_accounts": 2.6, "tech_savvy": True, "regions": ["Tashkent", "Samarkand", "Bukhara"]},
            "Transportation Business": {"income_weights": [0.35, 0.40, 0.20, 0.05], "avg_accounts": 2.4, "tech_savvy": False, "regions": ["Tashkent", "Andijan", "Fergana", "Nukus"]},
            "Real Estate Agent": {"income_weights": [0.20, 0.35, 0.35, 0.10], "avg_accounts": 2.7, "tech_savvy": True, "regions": ["Tashkent", "Samarkand", "Bukhara"]},
            "Student": {"income_weights": [0.85, 0.15, 0.00, 0.00], "avg_accounts": 1.2, "tech_savvy": True, "regions": ["Tashkent", "Samarkand", "Bukhara"]},
            "Retired": {"income_weights": [0.60, 0.35, 0.05, 0.00], "avg_accounts": 1.5, "tech_savvy": False, "regions": ["Tashkent", "Samarkand", "Bukhara", "Andijan", "Namangan", "Fergana", "Nukus"]}
        }

        client_profiles_cache = {}

        for client in clients:
            profile = occupation_profiles.get(client.occupation, occupation_profiles["Engineer"])

            # Pre-compute transaction type weights
            if client.occupation in ["Business Owner", "Import/Export Trader", "Restaurant Owner"]:
                type_weights = {"DEPOSIT": 0.35, "WITHDRAWAL": 0.20, "TRANSFER": 0.25, "PAYMENT": 0.15, "FEE": 0.05}
            elif client.occupation in ["Student", "Retired"]:
                type_weights = {"DEPOSIT": 0.25, "WITHDRAWAL": 0.40, "TRANSFER": 0.15, "PAYMENT": 0.15, "FEE": 0.05}
            else:
                type_weights = {"DEPOSIT": 0.30, "WITHDRAWAL": 0.30, "TRANSFER": 0.20, "PAYMENT": 0.15, "FEE": 0.05}

            # Pre-compute channel preferences
            if client.occupation == "Retired":
                channel_weights = {"BRANCH": 0.6, "ATM": 0.3, "ONLINE": 0.1}
            elif client.occupation == "Student":
                channel_weights = {"MOBILE": 0.4, "ONLINE": 0.3, "ATM": 0.2, "POS": 0.1}
            elif profile.get("tech_savvy", False):
                channel_weights = {"ONLINE": 0.4, "MOBILE": 0.3, "ATM": 0.2, "BRANCH": 0.1}
            else:
                channel_weights = {"ONLINE": 0.3, "ATM": 0.25, "BRANCH": 0.2, "MOBILE": 0.2, "POS": 0.05}

            # Pre-compute recurring payments
            recurring_payments = self._create_recurring_schedule_fast(client)

            client_profiles_cache[client.id] = {
                'occupation_profile': profile,
                'transaction_weights': type_weights,
                'channel_weights': channel_weights,
                'recurring_payments': recurring_payments,
                'tech_savvy': profile.get("tech_savvy", False),
                'occupation': client.occupation,
                'income_level': client.income_level,
                'risk_rating': client.risk_rating
            }

        logger.info(f"Pre-computed profiles for {len(clients)} clients")
        return client_profiles_cache

    def _create_recurring_schedule_fast(self, client) -> dict:
        """Optimized recurring payment schedule creation."""
        recurring_payments = {}

        # Monthly salary deposits
        if client.income_level != "LOW" or client.occupation != "Retired":
            if client.income_level == "ULTRA_HIGH":
                monthly_salary_sum = random.choice([150000, 180000, 200000, 220000, 250000])
            elif client.income_level == "HIGH":
                monthly_salary_sum = random.choice([80000, 100000, 120000, 150000])
            elif client.income_level == "MEDIUM":
                monthly_salary_sum = random.choice([30000, 40000, 50000, 60000, 70000])
            else:
                monthly_salary_sum = random.choice([15000, 20000, 25000, 30000])

            recurring_payments['SALARY'] = {
                'amount': sum_to_tiyin(monthly_salary_sum),
                'day_range': (25, 30),
                'type': 'DEPOSIT',
                'subtype': 'SALARY_DEPOSIT'
            }

        # Monthly utilities
        utility_amounts_sum = [2000, 2500, 3000, 3500, 4000]
        recurring_payments['UTILITIES'] = {
            'amount': sum_to_tiyin(random.choice(utility_amounts_sum)),
            'day_range': (1, 5),
            'type': 'PAYMENT',
            'subtype': 'UTILITY_PAYMENT'
        }

        return recurring_payments

    def _bulk_generate_dates(self, total_transactions: int, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Generate all transaction dates in bulk for massive performance improvement."""
        logger.info(f"Bulk generating {total_transactions} dates...")

        total_days = (end_date - start_date).days
        dates = []

        # Generate dates in large batches
        batch_size = 100000
        for batch_start in range(0, total_transactions, batch_size):
            batch_end = min(batch_start + batch_size, total_transactions)
            batch_size_actual = batch_end - batch_start

            # Vectorized date generation
            random_days = [random.randint(0, total_days) for _ in range(batch_size_actual)]

            batch_dates = []
            for days_offset in random_days:
                base_date = start_date + timedelta(days=days_offset)

                # Add time with business hour bias (70% during business hours)
                if random.random() < 0.7:  # Business hours
                    hour = random.randint(9, 17)
                else:
                    hour = random.choice([8, 18, 19, 20, 21, 22])

                minute = random.randint(0, 59)

                tx_date = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

                # Business day preference (move weekends to Monday)
                if tx_date.weekday() >= 5:  # Weekend
                    tx_date += timedelta(days=(7 - tx_date.weekday()))

                batch_dates.append(tx_date)

            dates.extend(batch_dates)

        # Sort all dates once for chronological order
        dates.sort()
        logger.info(f"Generated and sorted {len(dates)} dates")
        return dates

    def _create_fast_amount_lookup(self) -> dict:
        """Create lookup tables for fast amount generation."""
        return {
            # ATM withdrawal amounts by occupation (in tiyin)
            "ATM_WITHDRAWAL": {
                "Student": [sum_to_tiyin(x) for x in [500, 1000, 1500, 2000]],
                "Retired": [sum_to_tiyin(x) for x in [1000, 1500, 2000, 3000]],
                "Business Owner": [sum_to_tiyin(x) for x in [2000, 3000, 5000, 10000]],
                "IT Executive": [sum_to_tiyin(x) for x in [2000, 3000, 5000, 8000]],
                "Bank Executive": [sum_to_tiyin(x) for x in [3000, 5000, 8000, 10000]],
                "default": [sum_to_tiyin(x) for x in [1000, 2000, 3000, 5000]]
            },
            # Other withdrawal amounts by income level (in tiyin)
            "WITHDRAWAL": {
                "ULTRA_HIGH": [sum_to_tiyin(x) for x in [5000, 10000, 15000, 20000, 30000]],
                "HIGH": [sum_to_tiyin(x) for x in [3000, 5000, 8000, 10000, 15000]],
                "MEDIUM": [sum_to_tiyin(x) for x in [2000, 3000, 5000, 8000]],
                "LOW": [sum_to_tiyin(x) for x in [1000, 1500, 2000, 3000, 5000]]
            },
            # Payment amounts by occupation (in tiyin)
            "PAYMENT": {
                "Student": [sum_to_tiyin(x) for x in [500, 750, 1000, 1250, 1500]],
                "Retired": [sum_to_tiyin(x) for x in [800, 1200, 1500, 2000]],
                "default": [sum_to_tiyin(x) for x in [1500, 2000, 2500, 3000, 4000, 5000]]
            },
            # Cash deposit amounts by occupation (in tiyin)
            "CASH_DEPOSIT": {
                "Business Owner": [sum_to_tiyin(x) for x in [10000, 20000, 30000, 50000, 80000]],
                "Restaurant Owner": [sum_to_tiyin(x) for x in [10000, 20000, 30000, 50000, 80000]],
                "Transportation Business": [sum_to_tiyin(x) for x in [10000, 20000, 30000, 50000, 80000]],
                "default": [sum_to_tiyin(x) for x in [2000, 5000, 10000]]
            },
            # Transfer amounts (in tiyin)
            "TRANSFER": [sum_to_tiyin(x) for x in [5000, 10000, 15000, 20000, 30000, 50000]],
            # Fee amounts (in tiyin)
            "FEE": [sum_to_tiyin(x) for x in [100, 150, 250, 500, 750]]
        }

    def _generate_vectorized_transactions(self, total_transactions: int, dates: List[datetime],
                                        accounts_clients: List[tuple], profiles_cache: dict,
                                        amount_lookup: dict) -> None:
        """Generate transactions using vectorized operations for maximum performance."""

        logger.info("🚀 Starting vectorized transaction generation...")

        # Pre-compute account distribution to eliminate cycling
        accounts_data = []
        total_accounts = len(accounts_clients)
        transactions_per_account = total_transactions // total_accounts
        remaining_transactions = total_transactions % total_accounts

        account_idx = 0
        for account, client in accounts_clients:
            # Distribute transactions evenly with remainder
            account_transactions = transactions_per_account
            if account_idx < remaining_transactions:
                account_transactions += 1

            if account_transactions > 0:
                profile = profiles_cache[client.id]
                accounts_data.append({
                    'account': account,
                    'client': client,
                    'profile': profile,
                    'transaction_count': account_transactions,
                    'start_idx': sum(item['transaction_count'] for item in accounts_data[:-1]) if accounts_data else 0
                })

            account_idx += 1

        logger.info(f"Distributed {total_transactions} transactions across {len(accounts_data)} active accounts")

        # Process accounts in chunks for memory efficiency
        chunk_size = 1000  # Process 1000 accounts at a time
        transaction_counter = 1
        session = self.get_session()

        try:
            for chunk_start in range(0, len(accounts_data), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(accounts_data))
                chunk_accounts = accounts_data[chunk_start:chunk_end]

                # Calculate chunk transaction count
                chunk_transactions = sum(acc['transaction_count'] for acc in chunk_accounts)

                if chunk_transactions == 0:
                    continue

                logger.info(f"Processing account chunk {chunk_start//chunk_size + 1}/{(len(accounts_data)-1)//chunk_size + 1} ({chunk_transactions:,} transactions)")

                # Generate transactions for this chunk
                batch_transactions = []

                for acc_data in chunk_accounts:
                    account = acc_data['account']
                    client = acc_data['client']
                    profile = acc_data['profile']
                    start_idx = acc_data['start_idx']
                    tx_count = acc_data['transaction_count']

                    # Get date slice for this account
                    account_dates = dates[start_idx:start_idx + tx_count]

                    # Pre-generate random data for this account (vectorized)
                    tx_types_keys = list(profile['transaction_weights'].keys())
                    tx_types_weights = list(profile['transaction_weights'].values())
                    channel_keys = list(profile['channel_weights'].keys())
                    channel_weights = list(profile['channel_weights'].values())

                    # Vectorized random generation
                    random_tx_types = random.choices(tx_types_keys, weights=tx_types_weights, k=tx_count)
                    random_channels = random.choices(channel_keys, weights=channel_weights, k=tx_count)
                    random_fees = random.choices([0, 50, 100, 250], k=tx_count)

                    # Generate transactions for this account
                    for i in range(tx_count):
                        tx_date = account_dates[i]
                        transaction_type = random_tx_types[i]

                        # Fast subtype selection
                        subtypes = {
                            "DEPOSIT": ["SALARY_DEPOSIT", "CASH_DEPOSIT", "CHECK_DEPOSIT", "TRANSFER_IN"],
                            "WITHDRAWAL": ["ATM_WITHDRAWAL", "BRANCH_WITHDRAWAL", "ONLINE_WITHDRAWAL"],
                            "TRANSFER": ["INTERNAL_TRANSFER", "EXTERNAL_TRANSFER", "WIRE_TRANSFER"],
                            "PAYMENT": ["UTILITY_PAYMENT", "LOAN_PAYMENT", "MERCHANT_PAYMENT", "SUBSCRIPTION"],
                            "FEE": ["MONTHLY_FEE", "OVERDRAFT_FEE", "ATM_FEE", "WIRE_FEE"]
                        }
                        transaction_subtype = random.choice(subtypes[transaction_type])

                        # Smart channel override
                        if transaction_subtype == "ATM_WITHDRAWAL":
                            channel = "ATM"
                        elif transaction_subtype in ["CASH_DEPOSIT", "BRANCH_WITHDRAWAL"]:
                            channel = "BRANCH"
                        else:
                            channel = random_channels[i]

                        # Fast amount generation
                        amount = self._get_fast_amount(transaction_type, transaction_subtype,
                                                     profile, amount_lookup, tx_date)

                        # Simple risk scoring
                        risk_score = 0.0
                        if profile['risk_rating'] == "HIGH":
                            risk_score += 0.4
                        if abs(amount) > sum_to_tiyin(50000):
                            risk_score += 0.3
                        if tx_date.hour < 6 or tx_date.hour > 22:
                            risk_score += 0.1
                        risk_score = min(risk_score, 1.0)

                        # Create transaction
                        transaction_data = {
                            "transaction_reference": f"TXN{transaction_counter:015d}",
                            "account_id": account.id,
                            "transaction_type": transaction_type,
                            "transaction_subtype": transaction_subtype,
                            "amount": amount,
                            "fee_amount": sum_to_tiyin(random_fees[i]),
                            "channel": channel,
                            "balance_before": 0,
                            "balance_after": 0,
                            "risk_score": round(risk_score, 3),
                            "flagged_for_review": risk_score > 0.7,
                            "review_status": "CLEARED" if risk_score <= 0.7 else "PENDING",
                            "transaction_date": tx_date,
                            "processing_date": tx_date,
                            "created_by": "SYSTEM"
                        }

                        batch_transactions.append(transaction_data)
                        transaction_counter += 1

                # Insert chunk in smaller batches for memory efficiency
                insert_batch_size = 10000  # Smaller batches for reliability
                for batch_start in range(0, len(batch_transactions), insert_batch_size):
                    batch_end = min(batch_start + insert_batch_size, len(batch_transactions))
                    batch_to_insert = batch_transactions[batch_start:batch_end]

                    session.bulk_insert_mappings(Transaction, batch_to_insert)
                    session.commit()

                    logger.info(f"⚡ Generated {transaction_counter-1:,} transactions ({((transaction_counter-1)/total_transactions)*100:.1f}% complete)")

                # Clear memory
                del batch_transactions

        except Exception as e:
            logger.error(f"Error in vectorized transaction generation: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def _get_fast_amount(self, transaction_type: str, transaction_subtype: str,
                        profile: dict, amount_lookup: dict, tx_date: datetime) -> int:
        """Ultra-fast amount generation using lookup tables."""

        # Check recurring payments first
        for payment_type, schedule in profile['recurring_payments'].items():
            if (schedule['type'] == transaction_type and
                schedule['subtype'] == transaction_subtype and
                schedule['day_range'][0] <= tx_date.day <= schedule['day_range'][1]):

                if payment_type == 'UTILITIES':
                    # Seasonal variation
                    if tx_date.month in [12, 1, 2, 6, 7, 8]:
                        return int(schedule['amount'] * random.uniform(1.1, 1.3))
                    else:
                        return int(schedule['amount'] * random.uniform(0.9, 1.1))
                else:
                    return schedule['amount']

        # Use lookup tables for fast amount selection
        if transaction_type == "WITHDRAWAL":
            if transaction_subtype == "ATM_WITHDRAWAL":
                amounts = amount_lookup["ATM_WITHDRAWAL"].get(
                    profile['occupation'],
                    amount_lookup["ATM_WITHDRAWAL"]["default"]
                )
                return -random.choice(amounts)
            else:
                amounts = amount_lookup["WITHDRAWAL"][profile['income_level']]
                return -random.choice(amounts)

        elif transaction_type == "PAYMENT":
            if transaction_subtype == "MERCHANT_PAYMENT":
                # Use merchant category for realistic amounts
                return self._get_merchant_payment_amount(profile, tx_date)
            else:
                return -random.choice(amount_lookup["PAYMENT"]["default"])

        elif transaction_type == "DEPOSIT":
            if transaction_subtype == "CASH_DEPOSIT":
                amounts = amount_lookup["CASH_DEPOSIT"].get(
                    profile['occupation'],
                    amount_lookup["CASH_DEPOSIT"]["default"]
                )
                return random.choice(amounts)
            else:
                return sum_to_tiyin(random.randint(1000, 20000))

        elif transaction_type == "TRANSFER":
            amount = random.choice(amount_lookup["TRANSFER"])
            return amount if random.choice([True, False]) else -amount

        else:  # FEE
            return -random.choice(amount_lookup["FEE"])

    def _get_merchant_payment_amount(self, profile: dict, tx_date: datetime) -> int:
        """Generate realistic merchant payment amounts based on merchant categories."""
        from .config import settings

        # Select merchant category based on weights
        merchant_codes = list(settings.merchant_categories.keys())
        merchant_weights = [settings.merchant_categories[code]["weight"] for code in merchant_codes]
        selected_code = random.choices(merchant_codes, weights=merchant_weights)[0]

        merchant_info = settings.merchant_categories[selected_code]
        base_amount = merchant_info["avg_amount"]

        # Apply hour-based velocity patterns
        hour = tx_date.hour
        velocity_multiplier = 1.0
        for pattern_name, pattern_data in settings.transaction_velocity_patterns.items():
            if hour in pattern_data["hours"]:
                velocity_multiplier = pattern_data["multiplier"]
                break

        # Apply seasonal patterns
        seasonal_multiplier = 1.0
        for pattern_name, pattern_data in settings.seasonal_banking_cycles.items():
            if pattern_name == "SALARY_WEEKS" and tx_date.day in pattern_data["days"]:
                seasonal_multiplier = pattern_data.get("transaction_multiplier", 1.0)
            elif pattern_name == "MONTH_END" and tx_date.day in pattern_data["days"]:
                seasonal_multiplier = pattern_data.get("withdrawal_multiplier", 1.0)
            elif pattern_name == "HOLIDAY_SEASON" and tx_date.month in pattern_data["months"]:
                seasonal_multiplier = pattern_data["transaction_multiplier"]

        # Income level adjustment
        income_multipliers = {
            "LOW": 0.6,
            "MEDIUM": 1.0,
            "HIGH": 1.5,
            "ULTRA_HIGH": 2.0
        }
        income_multiplier = income_multipliers.get(profile.get('income_level', 'MEDIUM'), 1.0)

        # Calculate final amount with variance
        final_amount = base_amount * velocity_multiplier * seasonal_multiplier * income_multiplier
        # Add 20% variance
        variance = random.uniform(0.8, 1.2)
        final_amount = int(final_amount * variance)

        return -final_amount  # Negative for payment

    def _calculate_fast_risk(self, profile: dict, amount: int, tx_date: datetime, channel: str) -> float:
        """Fast risk calculation using simple rules."""
        risk_score = 0.0

        # High-risk client
        if profile['risk_rating'] == "HIGH":
            risk_score += 0.4

        # Large amounts
        if abs(amount) > sum_to_tiyin(50000):  # >50K сўм
            risk_score += 0.3

        # Unusual hours
        if tx_date.hour < 6 or tx_date.hour > 22:
            if channel in ["ATM", "ONLINE"] and abs(amount) > sum_to_tiyin(20000):
                risk_score += 0.15

        # Large cash deposits
        if amount > sum_to_tiyin(30000) and channel == "BRANCH":
            risk_score += 0.2

        return min(risk_score, 1.0)

    def _calculate_realistic_account_balance(self, client, account_type: str) -> int:
        """Calculate realistic account balance based on client profile and account type (returns tiyin)."""
        base_multiplier = {
            "ULTRA_HIGH": 50,
            "HIGH": 20,
            "MEDIUM": 8,
            "LOW": 3
        }

        # Account type multipliers for balance ranges (in сўм, will be converted to tiyin)
        type_ranges = {
            "CHECKING": (1000, 50000),      # 1K - 50K сўм
            "SAVINGS": (5000, 200000),      # 5K - 200K сўм
            "BUSINESS": (10000, 1000000),   # 10K - 1M сўм
            "CREDIT": (0, 20000),           # 0 - 20K сўм (credit balance)
            "DEPOSIT": (20000, 500000)      # 20K - 500K сўм (time deposits)
        }

        min_balance_sum, max_balance_sum = type_ranges.get(account_type, (1000, 50000))
        multiplier = base_multiplier.get(client.income_level, 3)

        # Adjust range based on income level
        adjusted_min = min_balance_sum * (multiplier / 10)
        adjusted_max = max_balance_sum * (multiplier / 10)

        # Generate random amount in сўм, then convert to tiyin
        sum_amount = random.uniform(adjusted_min, adjusted_max)
        return sum_to_tiyin(sum_amount)

    def _get_realistic_interest_rate(self, account_type: str, income_level: str) -> float:
        """Get realistic interest rates based on account type and client income level."""
        base_rates = {
            "CHECKING": 0.5,   # 0.5% annual
            "SAVINGS": 8.5,    # 8.5% annual (typical UZS rate)
            "BUSINESS": 6.0,   # 6.0% annual
            "CREDIT": 24.0,    # 24% annual (credit card rate)
            "DEPOSIT": 12.0    # 12% annual (time deposit)
        }

        # VIP clients get better rates
        income_adjustments = {
            "ULTRA_HIGH": 1.5,  # +1.5% better rate
            "HIGH": 0.8,        # +0.8% better rate
            "MEDIUM": 0.2,      # +0.2% better rate
            "LOW": 0.0          # No adjustment
        }

        base_rate = base_rates.get(account_type, 5.0)
        adjustment = income_adjustments.get(income_level, 0.0)

        # For credit accounts, higher income means lower interest rate
        if account_type == "CREDIT":
            return max(base_rate - adjustment, 18.0)  # Minimum 18% for credit
        else:
            return base_rate + adjustment

    def _get_minimum_balance_requirement(self, account_type: str) -> int:
        """Get minimum balance requirements by account type (returns tiyin)."""
        minimums_sum = {
            "CHECKING": 1000,       # 1K сўм
            "SAVINGS": 5000,        # 5K сўм
            "BUSINESS": 20000,      # 20K сўм
            "CREDIT": 0,            # No minimum for credit
            "DEPOSIT": 50000        # 50K сўм minimum for time deposits
        }

        sum_amount = minimums_sum.get(account_type, 1000)
        return sum_to_tiyin(sum_amount)

    # def _generate_central_database(self, regional_manager: RegionalDatabaseManager, force_regenerate: bool):
        """Generate central database with all reference data."""

        central_session = regional_manager.get_session('central')

        try:
            # Drop and recreate tables if force_regenerate
            if force_regenerate:
                Base.metadata.drop_all(regional_manager.engines['central'])
                logger.info("🗑️ Dropped existing central database tables")

            # Create tables in central database
            Base.metadata.create_all(regional_manager.engines['central'])
            logger.info("✅ Created central database tables")

            # Generate reference data (branches, products, clients, accounts)
            self._generate_reference_data(central_session)

            # Update file size info
            size_mb = regional_manager.get_file_size_mb('central')
            logger.info(f"📊 Central database: {size_mb:.1f}MB")

        except Exception as e:
            logger.error(f"❌ Error generating central database: {e}")
            central_session.rollback()
            raise
        finally:
            central_session.close()

    def _generate_reference_data(self, session):
        """Generate all reference data for central database."""

        fake = Faker()
        fake.seed_instance(42)
        random.seed(42)

        # Generate branches (16 branches across regions)
        logger.info("🏢 Generating branches...")
        branches_data = []
        for region in settings.regional_databases.keys():
            # 2-3 branches per region based on economic weight
            economic_weight = settings.regional_databases[region]['economic_weight']
            num_branches = max(1, int(economic_weight * 10))  # 1-3 branches per region

            for i in range(num_branches):
                branch_data = {
                    'branch_name': f"{region.title()} Branch {i+1}",
                    'branch_code': f"{region[:3].upper()}{i+1:03d}",
                    'city': region.title(),
                    'region': region.title(),
                    'branch_type': "FULL_SERVICE" if i == 0 else "STANDARD",
                    'status': 'ACTIVE',
                    'daily_cash_limit': sum_to_tiyin(100000000),  # 100M UZS
                    'max_transaction_amount': sum_to_tiyin(50000000),  # 50M UZS
                    'opened_date': datetime.now(),
                    'created_date': datetime.now(),
                    'created_by': 'SYSTEM'
                }
                branches_data.append(branch_data)

        session.bulk_insert_mappings(Branch, branches_data)
        session.commit()
        logger.info(f"✅ Generated {len(branches_data)} branches")

        # Generate products (banking products)
        logger.info("💳 Generating banking products...")
        products = [
            {
                'product_code': 'CHK001',
                'product_name': 'Premium Checking Account',
                'product_category': 'CHECKING',
                'product_type': 'CHECKING',
                'description': 'High-end checking with premium benefits',
                'base_interest_rate': 0.005,
                'annual_fee': sum_to_tiyin(6000),
                'minimum_balance': sum_to_tiyin(10000),
                'risk_category': 'LOW',
                'status': 'ACTIVE',
                'launch_date': datetime.now(),
                'created_date': datetime.now(),
                'created_by': 'SYSTEM'
            },
            {
                'product_code': 'SAV001',
                'product_name': 'Standard Savings Account',
                'product_category': 'SAVINGS',
                'product_type': 'SAVINGS',
                'description': 'Standard savings account with competitive rates',
                'base_interest_rate': 0.085,
                'annual_fee': sum_to_tiyin(2400),
                'minimum_balance': sum_to_tiyin(5000),
                'risk_category': 'LOW',
                'status': 'ACTIVE',
                'launch_date': datetime.now(),
                'created_date': datetime.now(),
                'created_by': 'SYSTEM'
            },
            {
                'product_code': 'BUS001',
                'product_name': 'Business Account',
                'product_category': 'BUSINESS',
                'product_type': 'BUSINESS',
                'description': 'Comprehensive business banking solution',
                'base_interest_rate': 0.06,
                'annual_fee': sum_to_tiyin(12000),
                'minimum_balance': sum_to_tiyin(20000),
                'risk_category': 'MEDIUM',
                'status': 'ACTIVE',
                'launch_date': datetime.now(),
                'created_date': datetime.now(),
                'created_by': 'SYSTEM'
            },
            {
                'product_code': 'STU001',
                'product_name': 'Student Account',
                'product_category': 'CHECKING',
                'product_type': 'STUDENT',
                'description': 'Special account for students with reduced fees',
                'base_interest_rate': 0.01,
                'annual_fee': 0,
                'minimum_balance': sum_to_tiyin(1000),
                'risk_category': 'LOW',
                'status': 'ACTIVE',
                'launch_date': datetime.now(),
                'created_date': datetime.now(),
                'created_by': 'SYSTEM'
            },
            {
                'product_code': 'DEP001',
                'product_name': 'Time Deposit',
                'product_category': 'DEPOSIT',
                'product_type': 'FIXED_DEPOSIT',
                'description': 'Fixed-term deposit with guaranteed returns',
                'base_interest_rate': 0.12,
                'annual_fee': 0,
                'minimum_balance': sum_to_tiyin(50000),
                'risk_category': 'LOW',
                'status': 'ACTIVE',
                'launch_date': datetime.now(),
                'created_date': datetime.now(),
                'created_by': 'SYSTEM'
            }
        ]

        session.bulk_insert_mappings(Product, products)
        session.commit()
        logger.info(f"✅ Generated {len(products)} banking products")

        # Generate clients with realistic regional distribution
        logger.info("👥 Generating clients with regional distribution...")
        all_clients = self._generate_regional_clients(session)

        # Generate accounts for clients
        logger.info("🏦 Generating accounts with regional banking patterns...")
        self._generate_regional_accounts(session, all_clients)

    def _generate_regional_clients(self, session) -> List[Client]:
        """Generate clients distributed across regions with realistic demographics."""

        fake = Faker()
        fake.seed_instance(42)

        total_clients = settings.num_clients
        all_clients = []

        for region, demographics in settings.regional_demographics.items():
            region_clients = int(total_clients * demographics['population_percent'])
            logger.info(f"  {region}: {region_clients} clients ({demographics['population_percent']*100:.1f}%)")

            # Get occupation weights for this region
            occupation_weights = demographics.get('occupation_weights', {})
            occupations = list(occupation_weights.keys())
            weights = list(occupation_weights.values())

            region_client_data = []
            for i in range(region_clients):
                # Realistic name generation for Uzbekistan
                first_name = fake.first_name()
                last_name = fake.last_name()

                # Age distribution realistic for banking
                age = random.choices(
                    [random.randint(18, 30), random.randint(31, 50), random.randint(51, 70), random.randint(71, 85)],
                    weights=[0.25, 0.45, 0.25, 0.05]  # Working age bias
                )[0]

                birth_date = datetime.now().date() - timedelta(days=age * 365 + random.randint(0, 365))

                # Select occupation based on regional weights
                occupation = random.choices(occupations, weights=weights)[0] if occupations else "Engineer"

                # Income level based on region's economic status and occupation
                gdp_multiplier = demographics['gdp_per_capita_multiplier']
                if occupation in ["Bank Executive", "IT Executive", "Business Owner"]:
                    income_weights = [0.05 * gdp_multiplier, 0.20 * gdp_multiplier, 0.50, 0.25]
                elif occupation in ["Doctor/Surgeon", "Government Official"]:
                    income_weights = [0.10, 0.40 * gdp_multiplier, 0.40, 0.10]
                elif occupation in ["Student", "Retired"]:
                    income_weights = [0.70, 0.25, 0.05, 0.00]
                else:
                    income_weights = [0.30, 0.50, 0.18, 0.02]

                # Normalize weights
                total_weight = sum(income_weights)
                income_weights = [w/total_weight for w in income_weights]

                income_level = random.choices(
                    ["LOW", "MEDIUM", "HIGH", "ULTRA_HIGH"],
                    weights=income_weights
                )[0]

                # Risk rating based on occupation and region
                if occupation in ["Government Official", "Teacher/Professor", "Bank Executive"]:
                    risk_rating = "LOW"
                elif occupation in ["Business Owner", "Import/Export Trader"] and region == "tashkent":
                    risk_rating = random.choices(["MEDIUM", "HIGH"], weights=[0.8, 0.2])[0]
                else:
                    risk_rating = random.choices(["LOW", "MEDIUM"], weights=[0.8, 0.2])[0]

                client_data = {
                    'client_number': f"CLT{region[:3].upper()}{i+1:06d}",
                    'name': f"{first_name} {last_name}",
                    'birth_date': birth_date,
                    'email': f"{first_name.lower()}.{last_name.lower()}@{random.choice(['gmail.com', 'mail.ru', 'yandex.com'])}",
                    'region': region.title(),
                    'occupation': occupation,
                    'income_level': income_level,
                    'risk_rating': risk_rating,
                    'status': 'ACTIVE',
                    'kyc_status': random.choices(["VERIFIED", "PENDING"], weights=[0.95, 0.05])[0],
                    'created_date': fake.date_time_between(start_date="-3y", end_date="-6m"),
                    'created_by': 'SYSTEM'
                }

                region_client_data.append(client_data)

            # Bulk insert regional clients
            session.bulk_insert_mappings(Client, region_client_data)
            session.commit()

            # Get the created clients for account generation
            region_clients_objects = session.query(Client).filter(Client.region == region.title()).all()
            all_clients.extend(region_clients_objects)

        logger.info(f"✅ Generated {len(all_clients)} clients across all regions")
        return all_clients

    def _generate_regional_accounts(self, session, all_clients):
        """Generate accounts for clients with regional banking patterns."""

        # Generate accounts (simplified for this implementation)
        accounts_data = []
        products = session.query(Product).all()
        branches = session.query(Branch).all()

        if not branches:
            logger.error("No branches found! Cannot generate accounts without branches.")
            return

        for client in all_clients:
            # Number of accounts based on income level and region
            if client.income_level == "ULTRA_HIGH":
                num_accounts = random.choices([2, 3, 4], weights=[0.3, 0.5, 0.2])[0]
            elif client.income_level == "HIGH":
                num_accounts = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
            else:
                num_accounts = random.choices([1, 2], weights=[0.7, 0.3])[0]

            # Get branches in the client's region (prefer local branches)
            region_branches = [b for b in branches if b.region == client.region]
            if not region_branches:
                region_branches = branches  # Fallback to any branch

            for i in range(num_accounts):
                # Account type selection based on client profile
                if client.occupation == "Student":
                    account_type = "CHECKING"
                elif client.occupation == "Business Owner":
                    account_type = random.choice(["BUSINESS", "CHECKING", "SAVINGS"])
                else:
                    account_type = random.choice(["CHECKING", "SAVINGS"])

                # Select a branch for this account (prefer main branch for first account)
                if i == 0 and len(region_branches) > 1:
                    # For first account, prefer main/full service branch
                    main_branches = [b for b in region_branches if b.branch_type == "FULL_SERVICE"]
                    selected_branch = random.choice(main_branches) if main_branches else random.choice(region_branches)
                else:
                    selected_branch = random.choice(region_branches)

                account_data = {
                    'client_id': client.id,
                    'branch_id': selected_branch.id,
                    'account_number': f"{random.randint(1000000000, 9999999999):010d}",
                    'account_type': account_type,
                    'balance': self._calculate_realistic_account_balance(client, account_type),
                    'available_balance': 0,  # Will be calculated
                    'currency': 'UZS',
                    'status': 'ACTIVE',
                    'open_date': fake.date_time_between(start_date="-5y", end_date="-6m"),
                    'created_date': datetime.utcnow()
                }
                account_data['available_balance'] = int(account_data['balance'] * 0.95)
                accounts_data.append(account_data)

        session.bulk_insert_mappings(Account, accounts_data)
        session.commit()
        logger.info(f"✅ Generated {len(accounts_data)} accounts")

    # def _generate_regional_database(self, region: str, regional_manager: RegionalDatabaseManager):
        """Generate transactions for a specific region with economic modeling."""

        logger.info(f"🏛️ Generating {region} database...")

        # Drop and recreate regional database tables
        Base.metadata.drop_all(regional_manager.engines[region])
        Base.metadata.create_all(regional_manager.engines[region])

        # Get configuration for this region
        config = settings.regional_databases[region]
        demographics = settings.regional_demographics[region]
        max_transactions = config['max_transactions']

        logger.info(f"  Target: {max_transactions:,} transactions (max file size ~{max_transactions*0.27/1000:.0f}MB)")

        # Generate transactions for this region using ultra-fast vectorized approach
        regional_session = regional_manager.get_session(region)

        try:
            # Get central database connection to fetch accounts for this region
            central_session = regional_manager.get_session('central')
            region_clients = central_session.query(Client).filter(Client.region == region.title()).all()
            region_accounts = []
            region_client_profiles = {}

            region_account_mapping = {}  # account_id -> client_id mapping
            for client in region_clients:
                accounts = central_session.query(Account).filter(Account.client_id == client.id).all()
                for account in accounts:
                    region_accounts.append(account.id)
                    region_account_mapping[account.id] = client.id

                # Build client profile for this region
                region_client_profiles[client.id] = {
                    'occupation': client.occupation,
                    'income_level': client.income_level or 'MEDIUM',  # Use existing income_level from model
                    'risk_rating': client.risk_rating,
                    'region': region,
                    'digital_adoption': demographics['digital_adoption']
                }
            central_session.close()

            if not region_accounts:
                logger.warning(f"No accounts found for region {region}")
                return

            logger.info(f"  Found {len(region_accounts)} accounts for {len(region_client_profiles)} clients")

            # Use ultra-fast vectorized generation adapted for regional patterns
            self._generate_regional_transactions_vectorized(
                regional_session, region, region_accounts, region_account_mapping, region_client_profiles,
                max_transactions, demographics
            )

            regional_session.commit()

            # Monitor file size
            db_path = Path(settings.database_dir) / config['file']
            if db_path.exists():
                size_mb = db_path.stat().st_size / (1024 * 1024)
                logger.info(f"  ✅ {region} database: {size_mb:.1f}MB - {'✓ OK' if size_mb < 95 else '❌ TOO BIG'}")

        except Exception as e:
            regional_session.rollback()
            logger.error(f"Error generating {region} database: {e}")
            raise
        finally:
            regional_session.close()

    def _generate_regional_transactions_vectorized(self, session, region: str, accounts: list,
                                                 account_to_client: dict, client_profiles: dict, target_count: int, demographics: dict):
        """Ultra-fast vectorized transaction generation for regional patterns."""

        logger.info(f"  🚀 Vectorized generation for {region} - targeting {target_count:,} transactions")

        # Regional-specific transaction patterns
        regional_patterns = self._get_regional_transaction_patterns(region, demographics)

        # Pre-compute all random selections for maximum speed
        batch_size = 50000
        total_batches = (target_count + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, target_count)
            current_batch_size = batch_end - batch_start

            if batch_num % 5 == 0:  # Progress logging
                logger.info(f"    Processing batch {batch_num + 1}/{total_batches} ({batch_start:,}-{batch_end:,})")

            # Vectorized random generation
            account_ids = np.random.choice(accounts, current_batch_size)

            # Regional transaction type distribution
            transaction_types = np.random.choice(
                list(regional_patterns['transaction_types'].keys()),
                current_batch_size,
                p=list(regional_patterns['transaction_types'].values())
            )

            # Regional channel preferences
            channels = np.random.choice(
                list(regional_patterns['channels'].keys()),
                current_batch_size,
                p=list(regional_patterns['channels'].values())
            )

            # Generate transaction references
            start_ref = batch_start
            references = [f"TXN{region.upper()}{(start_ref + i):010d}" for i in range(current_batch_size)]

            # Generate dates with regional seasonal patterns
            dates = self._generate_regional_dates_vectorized(current_batch_size, region)

            # Build batch transactions
            transactions_data = []
            for i in range(current_batch_size):
                account_id = int(account_ids[i])
                client_id = account_to_client.get(account_id)
                if not client_id:
                    continue

                profile = client_profiles.get(client_id, {})
                tx_type = transaction_types[i]
                channel = channels[i]

                # Generate transaction subtype based on regional patterns
                subtype = self._get_regional_subtype(tx_type, region, demographics)

                # Generate amount based on regional economic patterns
                amount = self._get_regional_amount(tx_type, subtype, profile, region, demographics)

                # Calculate fee
                fee = self._calculate_regional_fee(tx_type, channel, amount, region)

                # Risk score
                risk_score = self._calculate_fast_risk(profile, amount, dates[i], channel)

                transaction_data = {
                    'transaction_reference': references[i],
                    'account_id': account_id,
                    'transaction_type': tx_type,
                    'transaction_subtype': subtype,
                    'amount': amount,
                    'fee_amount': fee,
                    'currency': 'UZS',
                    'channel': channel,
                    'transaction_date': dates[i],
                    'balance_before': sum_to_tiyin(random.randint(10000, 100000)),
                    'balance_after': sum_to_tiyin(random.randint(10000, 100000)),
                    'risk_score': risk_score,
                    'flagged_for_review': risk_score > settings.risk_score_threshold,
                    'merchant_name': self._get_merchant_name(tx_type, subtype, region),
                    'merchant_category': self._get_merchant_category(tx_type, subtype),
                    'location': f"{region.title()}, Uzbekistan",
                    'description': f"{tx_type} via {channel}",
                    'status': 'COMPLETED'
                }

                transactions_data.append(transaction_data)

            # Bulk insert batch
            if transactions_data:
                session.bulk_insert_mappings(Transaction, transactions_data)
                session.commit()

        logger.info(f"  ✅ Generated {target_count:,} transactions for {region}")

    def _get_regional_transaction_patterns(self, region: str, demographics: dict) -> dict:
        """Get region-specific transaction type and channel distributions."""

        # Base patterns adjusted for regional characteristics
        base_patterns = {
            'transaction_types': {
                'DEPOSIT': 0.25,
                'WITHDRAWAL': 0.35,
                'TRANSFER': 0.25,
                'PAYMENT': 0.15
            },
            'channels': {
                'BRANCH': 0.30,
                'ATM': 0.25,
                'ONLINE': 0.25,
                'MOBILE': 0.20
            }
        }

        # Adjust based on digital adoption rate
        digital_adoption = demographics['digital_adoption']

        # Lower digital adoption means more branch/ATM usage
        if digital_adoption < 0.5:
            base_patterns['channels'] = {
                'BRANCH': 0.45,
                'ATM': 0.35,
                'ONLINE': 0.15,
                'MOBILE': 0.05
            }
        elif digital_adoption > 0.7:
            base_patterns['channels'] = {
                'BRANCH': 0.15,
                'ATM': 0.20,
                'ONLINE': 0.35,
                'MOBILE': 0.30
            }

        # Regional-specific adjustments
        if region == 'tashkent':
            # More digital payments in capital
            base_patterns['transaction_types']['PAYMENT'] = 0.25
            base_patterns['transaction_types']['WITHDRAWAL'] = 0.30
        elif region in ['andijan', 'fergana']:
            # More cash-based in agricultural regions
            base_patterns['transaction_types']['WITHDRAWAL'] = 0.40
            base_patterns['transaction_types']['DEPOSIT'] = 0.30

        # Normalize probabilities to ensure they sum to 1.0
        for category in ['transaction_types', 'channels']:
            total = sum(base_patterns[category].values())
            if total != 1.0:
                for key in base_patterns[category]:
                    base_patterns[category][key] /= total

        return base_patterns

    def _get_regional_subtype(self, tx_type: str, region: str, demographics: dict) -> str:
        """Get transaction subtype based on regional patterns."""

        subtypes = {
            'DEPOSIT': ['CASH_DEPOSIT', 'SALARY_DEPOSIT', 'TRANSFER_DEPOSIT', 'INTEREST_DEPOSIT'],
            'WITHDRAWAL': ['ATM_WITHDRAWAL', 'BRANCH_WITHDRAWAL', 'ONLINE_WITHDRAWAL'],
            'TRANSFER': ['ONLINE_TRANSFER', 'SWIFT_TRANSFER', 'INTERBANK_TRANSFER', 'INTERNAL_TRANSFER'],
            'PAYMENT': ['UTILITY_PAYMENT', 'MERCHANT_PAYMENT', 'ONLINE_PAYMENT', 'BILL_PAYMENT']
        }

        # Regional preferences for subtypes
        if region in ['tashkent', 'samarkand'] and demographics['digital_adoption'] > 0.6:
            if tx_type == 'PAYMENT':
                return random.choice(['ONLINE_PAYMENT', 'MERCHANT_PAYMENT', 'UTILITY_PAYMENT'])
            elif tx_type == 'TRANSFER':
                return random.choice(['ONLINE_TRANSFER', 'INTERBANK_TRANSFER'])

        return random.choice(subtypes.get(tx_type, ['DEFAULT']))

    def _get_regional_amount(self, tx_type: str, subtype: str, profile: dict, region: str, demographics: dict) -> int:
        """Generate transaction amount based on regional economic patterns."""

        # Get GDP multiplier for this region
        gdp_multiplier = demographics.get('gdp_per_capita_multiplier', 1.0)

        # Base amounts in UZS
        base_amounts = {
            'WITHDRAWAL': {
                'ATM_WITHDRAWAL': [50000, 100000, 200000, 500000],
                'BRANCH_WITHDRAWAL': [100000, 500000, 1000000, 2000000],
                'ONLINE_WITHDRAWAL': [50000, 200000, 500000, 1000000]
            },
            'DEPOSIT': {
                'CASH_DEPOSIT': [100000, 500000, 1000000, 5000000],
                'SALARY_DEPOSIT': [3000000, 5000000, 8000000, 15000000],
                'TRANSFER_DEPOSIT': [100000, 500000, 1000000, 3000000],
                'INTEREST_DEPOSIT': [10000, 50000, 100000, 200000]
            },
            'TRANSFER': [100000, 500000, 1000000, 3000000],
            'PAYMENT': {
                'UTILITY_PAYMENT': [50000, 100000, 200000, 500000],
                'MERCHANT_PAYMENT': [25000, 100000, 300000, 800000],
                'ONLINE_PAYMENT': [25000, 100000, 300000, 800000],
                'BILL_PAYMENT': [50000, 100000, 200000, 500000]
            }
        }

        # Get base amount
        if tx_type in ['WITHDRAWAL', 'DEPOSIT', 'PAYMENT'] and isinstance(base_amounts[tx_type], dict):
            # Use appropriate fallback for each transaction type
            if tx_type == 'DEPOSIT':
                fallback_key = 'CASH_DEPOSIT'
            elif tx_type == 'WITHDRAWAL':
                fallback_key = 'ATM_WITHDRAWAL'
            else:  # PAYMENT
                fallback_key = 'UTILITY_PAYMENT'

            amounts = base_amounts[tx_type].get(subtype, base_amounts[tx_type][fallback_key])
        else:
            amounts = base_amounts.get(tx_type, [100000, 500000])

        base_amount = random.choice(amounts)

        # Apply regional economic multiplier
        adjusted_amount = int(base_amount * gdp_multiplier)

        # Convert to tiyin
        return sum_to_tiyin(adjusted_amount)

    def _calculate_regional_fee(self, tx_type: str, channel: str, amount: int, region: str) -> int:
        """Calculate fees based on regional banking policies."""

        # Base fee structure
        if tx_type == 'TRANSFER':
            if abs(amount) > sum_to_tiyin(1000000):  # >1M UZS
                return sum_to_tiyin(5000)
            return sum_to_tiyin(2500)
        elif tx_type == 'WITHDRAWAL' and channel == 'ATM':
            return sum_to_tiyin(1000)
        elif tx_type == 'PAYMENT':
            return sum_to_tiyin(500)

        return 0

    def _generate_regional_dates_vectorized(self, count: int, region: str) -> list:
        """Generate transaction dates with regional seasonal patterns."""

        # Base date range (last 2 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)

        # Generate random dates
        time_between = end_date - start_date
        days_between = time_between.days

        random_days = np.random.randint(0, days_between, count)
        dates = [start_date + timedelta(days=int(day)) for day in random_days]

        # Apply seasonal patterns for specific regions
        seasonal_config = settings.seasonal_patterns.get(region, {})

        if seasonal_config:
            # Adjust dates to favor certain months
            if 'tourism_months' in seasonal_config:
                # Increase probability of transactions in tourism months
                tourism_months = seasonal_config['tourism_months']
                multiplier = seasonal_config.get('transaction_multiplier', 1.2)

                # Apply seasonal boost (simplified)
                adjusted_dates = []
                for date in dates:
                    if date.month in tourism_months and random.random() < 0.3:
                        # Add additional transactions in tourism season
                        adjusted_dates.extend([date] * int(multiplier))
                    else:
                        adjusted_dates.append(date)

                dates = adjusted_dates[:count]  # Trim to original count

        return dates

    def _get_merchant_name(self, tx_type: str, subtype: str, region: str) -> str:
        """Generate realistic merchant names for the region."""

        if tx_type != 'PAYMENT':
            return ""

        regional_merchants = {
            'tashkent': ['Tashkent City Mall', 'Next Store', 'Safia Restaurant', 'UzAuto Motors', 'IT Park Cafe'],
            'samarkand': ['Registan Bazaar', 'Samarkand Hotel', 'Bibi-Khanym Restaurant', 'Tourist Center'],
            'bukhara': ['Lyab-i Hauz Cafe', 'Bukhara Carpets', 'Silk Road Shop', 'Heritage Hotel'],
            'andijan': ['Andijan Bazaar', 'Agricultural Supply Co', 'Fergana Valley Store'],
            'fergana': ['Fergana Market', 'Industrial Supply', 'Kokand Textiles'],
            'namangan': ['Namangan Center', 'Regional Market', 'Education Books'],
            'nukus': ['Nukus Mall', 'Karakalpakstan Store', 'Regional Office']
        }

        merchants = regional_merchants.get(region, ['Local Store', 'Regional Market'])
        return random.choice(merchants)

    def _get_merchant_category(self, tx_type: str, subtype: str) -> str:
        """Get merchant category for transaction."""

        if tx_type != 'PAYMENT':
            return ""

        categories = {
            'UTILITY_PAYMENT': 'UTILITIES',
            'MERCHANT_PAYMENT': random.choice(['RETAIL', 'RESTAURANT', 'FUEL', 'GROCERY']),
            'ONLINE_PAYMENT': 'ECOMMERCE',
            'BILL_PAYMENT': 'SERVICES'
        }

        return categories.get(subtype, 'OTHER')

    # def _add_cross_regional_patterns(self, regional_manager: RegionalDatabaseManager):
        """Add cross-regional transfer patterns (placeholder)."""
        logger.info("📍 Adding cross-regional patterns (placeholder)")
        # This would add realistic cross-regional transfers
        pass

    # def _validate_file_sizes(self, regional_manager: RegionalDatabaseManager):
        """Validate all files are under GitHub limits."""
        logger.info("📏 Validating GitHub file size compliance...")

        all_compliant = True
        for db_name, info in regional_manager.get_database_info().items():
            size_mb = info['size_mb']
            if size_mb > 95:  # 95MB safety margin
                logger.error(f"❌ {db_name}: {size_mb:.1f}MB exceeds 95MB limit!")
                all_compliant = False
            else:
                logger.info(f"✅ {db_name}: {size_mb:.1f}MB (compliant)")

        if all_compliant:
            logger.info("🎉 All databases comply with GitHub file size limits!")
        else:
            raise ValueError("Some databases exceed GitHub file size limits!")

    def generate_mock_data(self, force_regenerate: bool = True):
        """Generate mock banking data with realistic patterns - always fresh."""

        logger.info("Generating fresh database with optimized schema...")
        logger.info(f"Target: {settings.num_clients} clients, {settings.num_transactions} transactions")

        # Always regenerate for clean, consistent results
        force_regenerate = True

        fake = Faker()
        fake.seed_instance(42)  # For reproducible data
        random.seed(42)

        session = self.get_session()

        try:
            # Always drop and recreate tables for fresh, clean database
            logger.info("Recreating tables with optimized schema...")
            session.close()  # Close session before dropping tables
            Base.metadata.drop_all(bind=self.engine)
            Base.metadata.create_all(bind=self.engine)
            self._create_performance_indexes()
            session = self.get_session()  # Get new session

            # Generate branches first
            regions = ["Tashkent", "Samarkand", "Bukhara", "Andijan", "Namangan", "Fergana", "Nukus"]
            branches = []

            logger.info("Generating branches...")
            for i, region in enumerate(regions):
                # Main branch in each region
                branch = Branch(
                    branch_code=f"{region[:3].upper()}001",
                    branch_name=f"{region} Main Branch",
                    city=region,
                    region=region,
                    branch_type="FULL_SERVICE",
                    daily_cash_limit=sum_to_tiyin(1000000),
                    max_transaction_amount=sum_to_tiyin(5000000)
                )
                branches.append(branch)

                # Additional branches in major cities
                if region in ["Tashkent", "Samarkand", "Bukhara"]:
                    for j in range(2, 5):  # 3 additional branches
                        branch = Branch(
                            branch_code=f"{region[:3].upper()}{j:03d}",
                            branch_name=f"{region} Branch #{j}",
                            city=region,
                            region=region,
                            branch_type=random.choice(["FULL_SERVICE", "DIGITAL"]),
                            daily_cash_limit=sum_to_tiyin(500000),
                            max_transaction_amount=sum_to_tiyin(2500000)
                        )
                        branches.append(branch)

            session.add_all(branches)
            session.commit()
            branch_ids = [b.id for b in session.query(Branch).all()]

            # Generate products
            logger.info("Generating products...")
            products = [
                # Checking Accounts
                Product(
                    product_code="CHK001",
                    product_name="Standard Checking",
                    product_category="ACCOUNT",
                    product_type="CHECKING",
                    description="Basic checking account for daily transactions",
                    base_interest_rate=0.5,
                    annual_fee=sum_to_tiyin(5),
                    minimum_balance=sum_to_tiyin(100)
                ),
                Product(
                    product_code="CHK002",
                    product_name="Premium Checking",
                    product_category="ACCOUNT",
                    product_type="CHECKING",
                    description="Premium checking with higher limits and benefits",
                    base_interest_rate=1.0,
                    annual_fee=sum_to_tiyin(15),
                    minimum_balance=sum_to_tiyin(1000)
                ),
                Product(
                    product_code="SAV001",
                    product_name="Standard Savings",
                    product_category="ACCOUNT",
                    product_type="SAVINGS",
                    description="Basic savings account with competitive interest",
                    base_interest_rate=3.5,
                    minimum_balance=sum_to_tiyin(50)
                ),
                Product(
                    product_code="SAV002",
                    product_name="High Yield Savings",
                    product_category="ACCOUNT",
                    product_type="SAVINGS",
                    description="High interest savings for larger balances",
                    base_interest_rate=5.5,
                    minimum_balance=sum_to_tiyin(10000)
                ),
                Product(
                    product_code="BUS001",
                    product_name="Business Checking",
                    product_category="ACCOUNT",
                    product_type="BUSINESS",
                    description="Business checking account for companies",
                    base_interest_rate=1.5,
                    annual_fee=sum_to_tiyin(25),
                    minimum_balance=sum_to_tiyin(500)
                ),
                Product(
                    product_code="LON001",
                    product_name="Personal Loan",
                    product_category="LOAN",
                    product_type="PERSONAL",
                    description="Personal loan for various needs",
                    base_interest_rate=12.0,
                    minimum_balance=sum_to_tiyin(100000)
                ),
                Product(
                    product_code="LON002",
                    product_name="Mortgage Loan",
                    product_category="LOAN",
                    product_type="MORTGAGE",
                    description="Home mortgage with competitive rates",
                    base_interest_rate=8.5,
                    minimum_balance=sum_to_tiyin(5000000)
                )
            ]

            session.add_all(products)
            session.commit()

            # Generate clients with sophisticated business patterns and behavior profiles
            income_levels = ["LOW", "MEDIUM", "HIGH", "ULTRA_HIGH"]
            risk_ratings = ["LOW", "MEDIUM", "HIGH"]
            kyc_statuses = ["VERIFIED", "PENDING", "EXPIRED"]

            # Professional occupations with realistic income distributions and banking behavior
            occupation_profiles = {
                "IT Executive": {"income_weights": [0.05, 0.20, 0.50, 0.25], "avg_accounts": 2.5, "tech_savvy": True, "regions": ["Tashkent", "Samarkand"]},
                "Bank Executive": {"income_weights": [0.02, 0.15, 0.60, 0.23], "avg_accounts": 3.2, "tech_savvy": True, "regions": ["Tashkent", "Samarkand", "Bukhara"]},
                "Doctor/Surgeon": {"income_weights": [0.03, 0.25, 0.55, 0.17], "avg_accounts": 2.8, "tech_savvy": False, "regions": ["Tashkent", "Samarkand", "Andijan"]},
                "Business Owner": {"income_weights": [0.10, 0.25, 0.45, 0.20], "avg_accounts": 3.8, "tech_savvy": True, "regions": ["Tashkent", "Samarkand", "Bukhara"]},
                "Government Official": {"income_weights": [0.15, 0.45, 0.35, 0.05], "avg_accounts": 2.2, "tech_savvy": False, "regions": ["Tashkent", "Namangan", "Fergana"]},
                "Engineer": {"income_weights": [0.20, 0.50, 0.25, 0.05], "avg_accounts": 2.1, "tech_savvy": True, "regions": ["Tashkent", "Samarkand", "Andijan"]},
                "Teacher/Professor": {"income_weights": [0.40, 0.45, 0.13, 0.02], "avg_accounts": 1.8, "tech_savvy": False, "regions": ["Tashkent", "Samarkand", "Bukhara", "Namangan"]},
                "Retail Manager": {"income_weights": [0.25, 0.55, 0.18, 0.02], "avg_accounts": 2.0, "tech_savvy": True, "regions": ["Tashkent", "Samarkand", "Andijan", "Fergana"]},
                "Construction Contractor": {"income_weights": [0.30, 0.45, 0.20, 0.05], "avg_accounts": 2.3, "tech_savvy": False, "regions": ["Tashkent", "Bukhara", "Nukus"]},
                "Agriculture Specialist": {"income_weights": [0.45, 0.40, 0.13, 0.02], "avg_accounts": 1.9, "tech_savvy": False, "regions": ["Fergana", "Andijan", "Namangan", "Nukus"]},
                "Import/Export Trader": {"income_weights": [0.15, 0.30, 0.40, 0.15], "avg_accounts": 3.5, "tech_savvy": True, "regions": ["Tashkent", "Samarkand"]},
                "Restaurant Owner": {"income_weights": [0.25, 0.45, 0.25, 0.05], "avg_accounts": 2.6, "tech_savvy": True, "regions": ["Tashkent", "Samarkand", "Bukhara"]},
                "Transportation Business": {"income_weights": [0.35, 0.40, 0.20, 0.05], "avg_accounts": 2.4, "tech_savvy": False, "regions": ["Tashkent", "Andijan", "Fergana", "Nukus"]},
                "Real Estate Agent": {"income_weights": [0.20, 0.35, 0.35, 0.10], "avg_accounts": 2.7, "tech_savvy": True, "regions": ["Tashkent", "Samarkand", "Bukhara"]},
                "Student": {"income_weights": [0.85, 0.15, 0.00, 0.00], "avg_accounts": 1.2, "tech_savvy": True, "regions": ["Tashkent", "Samarkand", "Bukhara"]},
                "Retired": {"income_weights": [0.60, 0.35, 0.05, 0.00], "avg_accounts": 1.5, "tech_savvy": False, "regions": regions}
            }

            logger.info("Generating clients...")
            client_batch_size = 5000  # Larger batch size for clients
            client_counter = 1

            for batch_start in range(0, settings.num_clients, client_batch_size):
                batch_end = min(batch_start + client_batch_size, settings.num_clients)
                client_data = []

                for i in range(batch_start, batch_end):
                    birth_date = fake.date_of_birth(minimum_age=18, maximum_age=80)
                    age = (datetime.now().date() - birth_date).days // 365

                    # Select occupation based on realistic age distributions
                    if age < 25:
                        # Young adults - students, entry-level positions
                        occupation_choices = ["Student", "Engineer", "Teacher/Professor", "Retail Manager"]
                        weights = [0.6, 0.2, 0.1, 0.1]
                    elif age < 35:
                        # Career building phase
                        occupation_choices = ["IT Executive", "Engineer", "Doctor/Surgeon", "Business Owner", "Teacher/Professor", "Government Official"]
                        weights = [0.25, 0.25, 0.15, 0.15, 0.15, 0.05]
                    elif age < 50:
                        # Peak earning years - executive and business roles
                        occupation_choices = ["Bank Executive", "Business Owner", "IT Executive", "Doctor/Surgeon", "Import/Export Trader", "Real Estate Agent", "Restaurant Owner"]
                        weights = [0.15, 0.20, 0.15, 0.15, 0.12, 0.13, 0.10]
                    elif age < 65:
                        # Senior professional roles
                        occupation_choices = ["Bank Executive", "Business Owner", "Doctor/Surgeon", "Government Official", "Construction Contractor", "Transportation Business"]
                        weights = [0.20, 0.25, 0.20, 0.15, 0.10, 0.10]
                    else:
                        # Retirement age
                        occupation_choices = ["Retired", "Business Owner", "Doctor/Surgeon", "Agriculture Specialist"]
                        weights = [0.70, 0.15, 0.10, 0.05]

                    occupation = random.choices(occupation_choices, weights=weights)[0]
                    profile = occupation_profiles[occupation]

                    # Income level based on occupation profile (not just age)
                    income_level = random.choices(income_levels, weights=profile["income_weights"])[0]

                    # Region selection based on occupation preferences
                    region = random.choice(profile["regions"])

                    # Risk rating influenced by occupation and income
                    if income_level == "ULTRA_HIGH" or occupation in ["Bank Executive", "IT Executive"]:
                        risk_weights = [0.8, 0.18, 0.02]  # Lower risk for high-income professionals
                    elif occupation in ["Student", "Retired"] or income_level == "LOW":
                        risk_weights = [0.4, 0.5, 0.1]   # Moderate risk for students/retirees
                    else:
                        risk_weights = [0.6, 0.35, 0.05]  # Standard distribution

                    # KYC status influenced by income level and occupation
                    if income_level in ["HIGH", "ULTRA_HIGH"] or occupation in ["Bank Executive", "Doctor/Surgeon"]:
                        kyc_weights = [0.95, 0.04, 0.01]  # High-income professionals have better KYC compliance
                    else:
                        kyc_weights = [0.75, 0.20, 0.05]  # Standard distribution

                    # Status influenced by risk rating and occupation
                    if profile.get("tech_savvy", False) and age < 50:
                        status_weights = [0.92, 0.07, 0.01]  # Tech-savvy professionals more likely to be active
                    elif occupation == "Retired":
                        status_weights = [0.75, 0.22, 0.03]  # Retirees less active
                    else:
                        status_weights = [0.85, 0.12, 0.03]  # Standard distribution

                    client_dict = {
                        "client_number": f"CL{client_counter:08d}",
                        "name": fake.name(),
                        "birth_date": birth_date,
                        "email": fake.email(),
                        "region": region,
                        "occupation": occupation,
                        "income_level": income_level,
                        "risk_rating": random.choices(risk_ratings, weights=risk_weights)[0],
                        "status": random.choices(["ACTIVE", "INACTIVE", "SUSPENDED"], weights=status_weights)[0],
                        "kyc_status": random.choices(kyc_statuses, weights=kyc_weights)[0],
                        "created_date": datetime.utcnow(),
                        "created_by": "SYSTEM"
                    }
                    client_data.append(client_dict)
                    client_counter += 1

                # Use bulk insert for better performance
                session.bulk_insert_mappings(Client, client_data)
                session.commit()

                # Clear memory
                del client_data
                session.expunge_all()

                logger.info(f"Generated {batch_end} clients ({(batch_end/settings.num_clients)*100:.1f}% complete)")

            logger.info("Generating accounts with professional banking relationships...")
            # Generate accounts with realistic banking relationships
            clients = session.query(Client).all()
            accounts = []
            account_counter = 1
            account_types = ["CHECKING", "SAVINGS", "BUSINESS", "INVESTMENT"]

            for client in clients:
                # Get client's occupation profile for sophisticated banking relationships
                profile = occupation_profiles.get(client.occupation, occupation_profiles["Engineer"])  # Default fallback

                # Account count based on occupation profile and income level
                base_accounts = int(profile["avg_accounts"])
                if client.income_level == "ULTRA_HIGH":
                    num_accounts = max(base_accounts, random.randint(2, 5))
                elif client.income_level == "HIGH":
                    num_accounts = max(base_accounts, random.randint(2, 4))
                elif client.income_level == "MEDIUM":
                    num_accounts = random.randint(1, max(2, base_accounts))
                else:
                    num_accounts = random.randint(1, 2)

                # Sophisticated account type selection based on occupation and profile
                client_account_types = ["CHECKING"]  # Everyone needs checking

                if num_accounts > 1:
                    # Business owners and executives need business accounts
                    if client.occupation in ["Business Owner", "Bank Executive", "Import/Export Trader", "Restaurant Owner"]:
                        available_types = ["SAVINGS", "BUSINESS", "INVESTMENT"]
                        if "BUSINESS" not in client_account_types:
                            client_account_types.append("BUSINESS")

                    # High-income professionals get investment accounts
                    elif client.income_level in ["HIGH", "ULTRA_HIGH"] and client.occupation in ["IT Executive", "Doctor/Surgeon", "Real Estate Agent"]:
                        available_types = ["SAVINGS", "INVESTMENT"]

                    # Students get basic savings
                    elif client.occupation == "Student":
                        available_types = ["SAVINGS"]

                    # Retirees prefer savings and deposits
                    elif client.occupation == "Retired":
                        available_types = ["SAVINGS"]

                    # Everyone else gets standard options
                    else:
                        if client.income_level in ["HIGH", "ULTRA_HIGH"]:
                            available_types = ["SAVINGS", "INVESTMENT"]
                        else:
                            available_types = ["SAVINGS"]

                    # Add additional account types
                    remaining_slots = num_accounts - len(client_account_types)
                    if remaining_slots > 0 and available_types:
                        additional_types = random.sample(
                            available_types,
                            min(remaining_slots, len(available_types))
                        )
                        client_account_types.extend(additional_types)

                for acc_idx, account_type in enumerate(client_account_types[:num_accounts]):
                    # Calculate realistic starting balance based on client profile (in tiyin)
                    balance = self._calculate_realistic_account_balance(client, account_type)

                    account = Account(
                        account_number=f"AC{account_counter:012d}",
                        client_id=client.id,
                        branch_id=random.choice(branch_ids),
                        account_type=account_type,
                        balance=balance,
                        available_balance=int(balance * random.uniform(0.92, 0.99)),
                        interest_rate=self._get_realistic_interest_rate(account_type, client.income_level),
                        minimum_balance=self._get_minimum_balance_requirement(account_type),
                        open_date=fake.date_between(start_date="-5y", end_date="-6m"),
                        status=random.choices(["ACTIVE", "INACTIVE", "FROZEN"], weights=[0.94, 0.05, 0.01])[0]
                    )
                    accounts.append(account)
                    account_counter += 1

                    if len(accounts) >= 1000:
                        session.add_all(accounts)
                        session.commit()
                        accounts = []

            if accounts:
                session.add_all(accounts)
                session.commit()

            # ULTRA-FAST TRANSACTION GENERATION WITH SMART DATA PRESERVED
            logger.info("🚀 Starting ultra-performance transaction generation...")

            # Phase 1: Pre-computation (10-15 seconds)
            logger.info("Phase 1: Pre-computing client profiles and lookup tables...")
            clients_list = session.query(Client).all()
            profiles_cache = self._pre_compute_client_profiles(clients_list)
            amount_lookup = self._create_fast_amount_lookup()

            # Get account-client pairs for fast access
            accounts_with_clients = session.query(Account, Client).join(Client).all()
            logger.info(f"Processing {len(accounts_with_clients)} account-client pairs")

            # Phase 2: Bulk date generation (5-10 seconds)
            logger.info("Phase 2: Bulk generating chronological transaction dates...")
            start_date_base = datetime(2023, 1, 1)
            end_date_base = datetime(2025, 6, 30)
            all_dates = self._bulk_generate_dates(settings.num_transactions, start_date_base, end_date_base)

            # Phase 3: Vectorized transaction generation (lightning fast!)
            logger.info("Phase 3: Vectorized transaction generation...")
            self._generate_vectorized_transactions(
                settings.num_transactions, all_dates, accounts_with_clients,
                profiles_cache, amount_lookup
            )

            # Phase 4: Fast balance calculation with bulk operations
            logger.info("Phase 4: Calculating running balances with bulk operations...")
            # Simple random balances for demo - much faster than complex calculations
            account_updates = []
            for account, client in accounts_with_clients:
                final_balance = sum_to_tiyin(random.uniform(1000, 50000))  # Random realistic balance
                account_updates.append({
                    'id': account.id,
                    'balance': final_balance,
                    'available_balance': int(final_balance * 0.95)
                })

            logger.info(f"Bulk updating {len(account_updates):,} account balances...")
            # Use efficient bulk update operation instead of individual queries
            session.bulk_update_mappings(Account, account_updates)
            session.commit()
            logger.info("✅ Account balances updated successfully!")

            logger.info("🎉 Ultra-fast mock data generation completed with preserved smart features!")

            # Print some statistics
            stats = self.get_database_stats(session)
            logger.info(f"Database statistics: {stats}")

        except Exception as e:
            logger.error(f"Error generating mock data: {e}")
            session.rollback()
            raise
        finally:
            session.close()


    def get_database_stats(self, session: Optional[Session] = None) -> dict:
        """Get database statistics."""
        if session is None:
            session = self.get_session()
            close_session = True
        else:
            close_session = False

        try:
            stats = {
                "clients": session.query(Client).count(),
                "accounts": session.query(Account).count(),
                "transactions": session.query(Transaction).count(),
                "branches": session.query(Branch).count(),
                "products": session.query(Product).count(),
                "regions": [r[0] for r in session.query(Client.region.distinct())],
                "account_types": [r[0] for r in session.query(Account.account_type.distinct())],
                "transaction_types": [r[0] for r in session.query(Transaction.transaction_type.distinct())],
                "date_range": {
                    "first_transaction": session.query(func.min(Transaction.transaction_date)).scalar(),
                    "last_transaction": session.query(func.max(Transaction.transaction_date)).scalar(),
                }
            }
            return stats
        finally:
            if close_session:
                session.close()

    @contextmanager
    def get_session_context(self) -> Session:
        """Context manager for database sessions with automatic cleanup.

        Yields:
            Database session with automatic transaction management

        Raises:
            DatabaseConnectionError: If session creation fails
        """
        session = None
        try:
            session = self.SessionLocal()
            yield session
            session.commit()
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise DatabaseError(f"Database session error: {str(e)}")
        finally:
            if session:
                session.close()

    def execute_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute SQL query with comprehensive error handling and routing.

        Args:
            sql_query: SQL query string to execute

        Returns:
            List of dictionaries representing query results

        Raises:
            ValidationError: If query is invalid
            QueryExecutionError: If query execution fails
            DatabaseError: For other database-related errors
        """
        if not sql_query or not isinstance(sql_query, str):
            raise ValidationError(
                "SQL query must be a non-empty string",
                field_name="sql_query",
                invalid_value=str(sql_query)
            )

        sql_query = sql_query.strip()
        if not sql_query:
            raise ValidationError(
                "SQL query cannot be empty or whitespace only",
                field_name="sql_query"
            )

        logger.info(f"🔍 Executing query: {sql_query[:100]}...")

        # Execute query on single database with all regional data
        try:
            with self.get_session_context() as session:
                result = session.execute(text(sql_query))
                columns = list(result.keys())
                rows = result.fetchall()

                if not rows:
                    logger.info("✅ Query executed successfully - no results")
                    return []

                result_data = [dict(zip(columns, row)) for row in rows]
                logger.info(f"✅ Query executed successfully - {len(result_data)} rows returned")
                return result_data

        except SQLAlchemyError as e:
            error_msg = f"SQL execution error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            raise QueryExecutionError(
                error_msg,
                query=sql_query,
                database_name=self._mask_url(self.database_url)
            )
        except Exception as e:
            error_msg = f"Unexpected database error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            raise DatabaseError(
                error_msg,
                database_name=self._mask_url(self.database_url)
            )

    def get_database_context_for_llm(self) -> Dict[str, Any]:
        """Sample actual database content to provide context awareness for LLM.

        Returns:
            Dictionary containing real data samples and statistics for LLM context
        """
        try:
            with self.get_session_context() as session:
                context = {}

                # Sample branch data by region
                branches_sample = session.execute(text("""
                    SELECT region, branch_name, COUNT(*) as branch_count
                    FROM branches
                    GROUP BY region, branch_name
                    ORDER BY region, branch_name
                    LIMIT 20
                """)).fetchall()

                context["branches"] = {
                    "sample_names": [row[1] for row in branches_sample],
                    "regions": list(set(row[0] for row in branches_sample)),
                    "examples": [f"{row[1]} in {row[0]}" for row in branches_sample[:10]]
                }

                # Sample client data by region
                clients_sample = session.execute(text("""
                    SELECT region, COUNT(*) as client_count,
                           GROUP_CONCAT(name, ', ') as sample_names
                    FROM clients
                    GROUP BY region
                    ORDER BY client_count DESC
                """)).fetchall()

                context["clients"] = {
                    "by_region": {row[0]: row[1] for row in clients_sample},
                    "sample_names": clients_sample[0][2].split(', ')[:5] if clients_sample else [],
                    "total_clients": sum(row[1] for row in clients_sample)
                }

                # Sample account types and balances
                accounts_sample = session.execute(text("""
                    SELECT account_type, COUNT(*) as count,
                           AVG(balance)/100.0 as avg_balance_som,
                           MIN(balance)/100.0 as min_balance_som,
                           MAX(balance)/100.0 as max_balance_som
                    FROM accounts
                    GROUP BY account_type
                """)).fetchall()

                context["accounts"] = {
                    "types": [row[0] for row in accounts_sample],
                    "statistics": {row[0]: {
                        "count": row[1],
                        "avg_balance": round(row[2], 2),
                        "min_balance": round(row[3], 2),
                        "max_balance": round(row[4], 2)
                    } for row in accounts_sample}
                }

                # Sample transaction patterns
                transactions_sample = session.execute(text("""
                    SELECT transaction_type, COUNT(*) as count,
                           AVG(amount)/100.0 as avg_amount_som,
                           SUM(amount)/100.0 as total_volume_som
                    FROM transactions
                    GROUP BY transaction_type
                    ORDER BY count DESC
                    LIMIT 10
                """)).fetchall()

                context["transactions"] = {
                    "types": [row[0] for row in transactions_sample],
                    "statistics": {row[0]: {
                        "count": row[1],
                        "avg_amount": round(row[2], 2),
                        "total_volume": round(row[3], 2)
                    } for row in transactions_sample}
                }

                # Regional transaction distribution
                regional_stats = session.execute(text("""
                    SELECT c.region, COUNT(t.id) as transaction_count,
                           SUM(t.amount)/100.0 as total_volume_som
                    FROM transactions t
                    JOIN accounts a ON t.account_id = a.id
                    JOIN clients c ON a.client_id = c.id
                    GROUP BY c.region
                    ORDER BY transaction_count DESC
                """)).fetchall()

                context["regional_activity"] = {
                    row[0]: {
                        "transaction_count": row[1],
                        "total_volume": round(row[2], 2)
                    } for row in regional_stats
                }

                return context

        except Exception as e:
            logger.warning(f"Could not sample database content: {e}")
            return {
                "branches": {"regions": ["Tashkent", "Samarkand", "Bukhara", "Andijan", "Fergana", "Namangan", "Nukus"]},
                "clients": {"total_clients": 0},
                "accounts": {"types": ["CHECKING", "SAVINGS"]},
                "transactions": {"types": ["DEPOSIT", "WITHDRAWAL", "TRANSFER"]},
                "regional_activity": {}
            }



# Global database manager instance
db_manager = DatabaseManager()

logger.info("🚀 Banking system initialized with single database architecture")