"""Database models and operations for BankingLLM system."""

import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional

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
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session

from .config import settings

Base = declarative_base()

# Currency conversion helpers - 1 сўм = 100 тийин
def sum_to_tiyin(sum_amount: float) -> int:
    """Convert сўм amount to тийин (multiply by 100)."""
    return int(round(sum_amount * 100))

def tiyin_to_sum(tiyin_amount: int) -> float:
    """Convert тийин amount to сўм (divide by 100)."""
    return tiyin_amount / 100.0

def format_currency(tiyin_amount: int, language: str = "english", show_tiyin: bool = False) -> str:
    """Format tiyin amount as currency string with language-specific symbols."""
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

# Fake instance for location generation
fake = Faker()


class Branch(Base):
    """Branch model representing bank branches with location and operational details."""

    __tablename__ = "branches"

    id = Column(Integer, primary_key=True, index=True)
    branch_code = Column(String(10), unique=True, nullable=False, index=True)
    branch_name = Column(String(100), nullable=False)

    # Location Information
    address = Column(String(200), nullable=False)
    city = Column(String(50), nullable=False, index=True)
    region = Column(String(50), nullable=False, index=True)
    country = Column(String(50), nullable=False, default="Uzbekistan")
    postal_code = Column(String(10), nullable=True)

    # Contact Information
    phone = Column(String(20), nullable=True)
    email = Column(String(100), nullable=True)
    manager_name = Column(String(100), nullable=True)

    # Operational Details
    branch_type = Column(String(20), nullable=False, default="FULL_SERVICE", index=True)  # FULL_SERVICE, ATM_ONLY, DIGITAL, CORPORATE
    status = Column(String(20), nullable=False, default="ACTIVE", index=True)  # ACTIVE, INACTIVE, CLOSED
    operating_hours = Column(String(100), nullable=True)
    services = Column(String(500), nullable=True)  # Comma-separated services

    # Financial Limits (stored in tiyin)
    daily_cash_limit = Column(BigInteger, nullable=True)
    max_transaction_amount = Column(BigInteger, nullable=True)

    # Audit Fields
    opened_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    closed_date = Column(DateTime, nullable=True)
    created_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(50), nullable=False, default="SYSTEM")

    # Relationships
    accounts = relationship("Account", back_populates="branch")

    def __repr__(self):
        return f"<Branch(id={self.id}, code='{self.branch_code}', name='{self.branch_name}', city='{self.city}')>"


class Product(Base):
    """Product model representing banking products and services."""

    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    product_code = Column(String(20), unique=True, nullable=False, index=True)
    product_name = Column(String(100), nullable=False)
    product_category = Column(String(30), nullable=False, index=True)  # ACCOUNT, LOAN, CARD, INVESTMENT, INSURANCE
    product_type = Column(String(30), nullable=False, index=True)  # CHECKING, SAVINGS, MORTGAGE, CREDIT_CARD, etc.

    # Product Details
    description = Column(String(500), nullable=True)
    currency = Column(String(3), nullable=False, default="UZS")

    # Interest and Fees
    base_interest_rate = Column(Numeric(5, 4), nullable=True, default=0.0)  # Up to 99.9999% interest rate
    min_interest_rate = Column(Numeric(5, 4), nullable=True)
    max_interest_rate = Column(Numeric(5, 4), nullable=True)
    annual_fee = Column(BigInteger, nullable=True, default=0)  # stored in tiyin
    maintenance_fee = Column(BigInteger, nullable=True, default=0)  # stored in tiyin

    # Limits and Requirements (stored in tiyin)
    minimum_balance = Column(BigInteger, nullable=True, default=0)
    maximum_balance = Column(BigInteger, nullable=True)
    minimum_deposit = Column(BigInteger, nullable=True)
    credit_limit = Column(BigInteger, nullable=True)

    # Eligibility and Risk
    minimum_age = Column(Integer, nullable=True, default=18)
    maximum_age = Column(Integer, nullable=True)
    minimum_income = Column(BigInteger, nullable=True)  # stored in tiyin
    credit_score_required = Column(Integer, nullable=True)
    risk_category = Column(String(20), nullable=False, default="MEDIUM", index=True)

    # Product Status
    status = Column(String(20), nullable=False, default="ACTIVE", index=True)  # ACTIVE, DISCONTINUED, SUSPENDED
    launch_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    discontinue_date = Column(DateTime, nullable=True)

    # Marketing and Terms
    promotional_rate = Column(Numeric(5, 4), nullable=True)
    promotional_period_months = Column(Integer, nullable=True)
    terms_and_conditions = Column(String(1000), nullable=True)

    # Audit Fields
    created_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(50), nullable=False, default="SYSTEM")

    def __repr__(self):
        return f"<Product(id={self.id}, code='{self.product_code}', name='{self.product_name}', category='{self.product_category}')>"


class Client(Base):
    """Client model representing bank customers with professional banking fields."""

    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    client_number = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    birth_date = Column(DateTime, nullable=False)

    # Contact Information
    email = Column(String(100), nullable=True, index=True)
    phone = Column(String(20), nullable=True)

    # Address Information
    address = Column(String(200), nullable=True)
    city = Column(String(50), nullable=True)
    region = Column(String(50), nullable=False, index=True)
    country = Column(String(50), nullable=False, default="Uzbekistan")
    postal_code = Column(String(10), nullable=True)

    # KYC/AML Information
    occupation = Column(String(100), nullable=True)
    income_level = Column(String(20), nullable=True, index=True)  # LOW, MEDIUM, HIGH, ULTRA_HIGH
    risk_rating = Column(String(10), nullable=False, default="MEDIUM", index=True)  # LOW, MEDIUM, HIGH

    # Account Status and Compliance
    status = Column(String(20), nullable=False, default="ACTIVE", index=True)  # ACTIVE, INACTIVE, SUSPENDED, CLOSED
    kyc_status = Column(String(20), nullable=False, default="PENDING")  # PENDING, VERIFIED, EXPIRED

    # Audit Fields
    created_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(50), nullable=False, default="SYSTEM")

    # Relationships
    accounts = relationship("Account", back_populates="client", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Client(id={self.id}, client_number='{self.client_number}', name='{self.name}', region='{self.region}')>"


class Account(Base):
    """Account model representing bank accounts with professional banking features."""

    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    account_number = Column(String(20), unique=True, nullable=False, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False, index=True)
    branch_id = Column(Integer, ForeignKey("branches.id"), nullable=False, index=True)

    # Account Details
    account_type = Column(String(20), nullable=False, index=True)  # CHECKING, SAVINGS, BUSINESS, INVESTMENT, LOAN
    account_subtype = Column(String(30), nullable=True)  # STANDARD_CHECKING, PREMIUM_SAVINGS, etc.
    currency = Column(String(3), nullable=False, default="UZS", index=True)  # UZS, USD, EUR

    # Balances and Limits (stored in tiyin for exact precision)
    balance = Column(BigInteger, nullable=False, default=0, index=True)
    available_balance = Column(BigInteger, nullable=False, default=0)
    overdraft_limit = Column(BigInteger, nullable=True, default=0)
    daily_transaction_limit = Column(BigInteger, nullable=True)

    # Interest and Fees
    interest_rate = Column(Numeric(5, 4), nullable=True, default=0.0)
    monthly_fee = Column(BigInteger, nullable=True, default=0)  # stored in tiyin
    minimum_balance = Column(BigInteger, nullable=True, default=0)  # stored in tiyin

    # Account Status
    status = Column(String(20), nullable=False, default="ACTIVE", index=True)  # ACTIVE, INACTIVE, CLOSED, FROZEN
    close_reason = Column(String(100), nullable=True)

    # Audit Fields
    open_date = Column(DateTime, nullable=False, index=True)
    close_date = Column(DateTime, nullable=True)
    last_transaction_date = Column(DateTime, nullable=True, index=True)
    created_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(50), nullable=False, default="SYSTEM")

    # Relationships
    client = relationship("Client", back_populates="accounts")
    branch = relationship("Branch", back_populates="accounts")
    transactions = relationship("Transaction", back_populates="account", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Account(id={self.id}, account_number='{self.account_number}', type='{self.account_type}', balance={self.balance})>"


class Transaction(Base):
    """Transaction model representing bank transactions with comprehensive banking details."""

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_reference = Column(String(30), unique=True, nullable=False, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False, index=True)
    sequence_number = Column(Integer, nullable=False, index=True)  # Sequential number per account

    # Transaction Core Details
    transaction_type = Column(String(30), nullable=False, index=True)  # DEPOSIT, WITHDRAWAL, TRANSFER, PAYMENT, FEE
    transaction_subtype = Column(String(50), nullable=True)  # SALARY_DEPOSIT, ATM_WITHDRAWAL, LOAN_PAYMENT
    amount = Column(BigInteger, nullable=False, index=True)  # stored in tiyin for exact precision
    currency = Column(String(3), nullable=False, default="UZS")

    # Exchange and Fees
    original_amount = Column(BigInteger, nullable=True)  # For currency conversions, stored in tiyin
    original_currency = Column(String(3), nullable=True)
    exchange_rate = Column(Numeric(10, 6), nullable=True)  # More precision for exchange rates
    fee_amount = Column(BigInteger, nullable=False, default=0)  # stored in tiyin

    # Transaction Context
    description = Column(String(200), nullable=True)
    merchant_name = Column(String(100), nullable=True)
    merchant_category = Column(String(50), nullable=True)
    location = Column(String(100), nullable=True)

    # Channel and Processing
    channel = Column(String(20), nullable=False, index=True)  # ATM, ONLINE, BRANCH, MOBILE, POS
    device_id = Column(String(50), nullable=True)
    ip_address = Column(String(45), nullable=True)

    # Balances and Status (stored in tiyin for exact precision)
    balance_before = Column(BigInteger, nullable=False)
    balance_after = Column(BigInteger, nullable=False)
    available_balance_after = Column(BigInteger, nullable=False)
    status = Column(String(20), nullable=False, default="COMPLETED", index=True)  # PENDING, COMPLETED, FAILED, CANCELLED

    # Related Transactions (for transfers)
    related_account_id = Column(Integer, nullable=True)  # For transfers
    related_transaction_id = Column(Integer, nullable=True)

    # Risk and Compliance
    risk_score = Column(Numeric(3, 3), nullable=True)  # 0.000 to 0.999 risk score
    flagged_for_review = Column(Boolean, default=False, index=True)
    review_status = Column(String(20), nullable=True)  # CLEARED, SUSPICIOUS, BLOCKED

    # Audit Fields
    transaction_date = Column(DateTime, nullable=False, index=True)
    processing_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    value_date = Column(DateTime, nullable=False, index=True)
    created_by = Column(String(50), nullable=False, default="SYSTEM")
    authorized_by = Column(String(50), nullable=True)

    # Relationships
    account = relationship("Account", back_populates="transactions")

    def __repr__(self):
        return f"<Transaction(id={self.id}, ref='{self.transaction_reference}', type='{self.transaction_type}', amount={self.amount})>"


class DatabaseManager:
    """Database manager for creating connections and managing data."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database_url

        # Optimized engine configuration with SQLite performance pragmas
        connect_args = {}
        if "sqlite" in self.database_url:
            connect_args = {
                "check_same_thread": False,
                # SQLite performance optimizations
                "timeout": 30.0,
            }

        self.engine = create_engine(
            self.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=10,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=settings.log_level == "DEBUG",  # SQL logging in debug mode
            connect_args=connect_args
        )

        # Apply SQLite performance pragmas
        if "sqlite" in self.database_url:
            self._optimize_sqlite()

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            expire_on_commit=False  # Performance optimization
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

    def create_tables(self):
        """Create all database tables."""
        logger.info("Creating database tables...")
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

                # Account analysis indexes with constraints
                "CREATE INDEX IF NOT EXISTS idx_account_type_currency_balance ON accounts(account_type, currency, balance)",
                "CREATE INDEX IF NOT EXISTS idx_account_client_status ON accounts(client_id, status)",
                "CREATE INDEX IF NOT EXISTS idx_account_branch_type_balance ON accounts(branch_id, account_type, balance)",
                "CREATE INDEX IF NOT EXISTS idx_account_currency_status ON accounts(currency, status)",
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

    def generate_mock_data(self, force_regenerate: bool = False):
        """Generate mock banking data with realistic patterns."""

        # Check if database already contains data (protection against accidental regeneration)
        if not force_regenerate and self._database_has_data():
            logger.warning("🛡️  DATABASE PROTECTION: Database already contains data!")
            logger.warning("🛡️  This sophisticated 1M+ record database took significant time to generate.")
            logger.warning("🛡️  To regenerate anyway, use --force-regenerate flag or force_regenerate=True")
            logger.warning("🛡️  Current database stats:")

            # Show current database statistics
            stats = self.get_database_stats()
            for table, count in stats.items():
                if isinstance(count, int):
                    logger.warning(f"🛡️    {table}: {count:,} records")

            return  # Exit without regenerating

        if force_regenerate:
            logger.warning("⚠️  FORCE REGENERATION: Proceeding to regenerate database as requested")
            logger.warning("⚠️  This will delete all existing sophisticated banking data!")

        logger.info(f"Generating mock data: {settings.num_clients} clients, {settings.num_transactions} transactions")

        fake = Faker()
        fake.seed_instance(42)  # For reproducible data
        random.seed(42)

        session = self.get_session()

        try:
            # Clear existing data in correct order (foreign keys)
            session.execute(text("DELETE FROM transactions"))
            session.execute(text("DELETE FROM accounts"))
            session.execute(text("DELETE FROM clients"))
            session.execute(text("DELETE FROM products"))
            session.execute(text("DELETE FROM branches"))
            session.commit()

            # Generate branches first
            regions = ["Tashkent", "Samarkand", "Bukhara", "Andijan", "Namangan", "Fergana", "Nukus"]
            branches = []

            logger.info("Generating branches...")
            for i, region in enumerate(regions):
                # Main branch in each region
                branch = Branch(
                    branch_code=f"{region[:3].upper()}001",
                    branch_name=f"{region} Main Branch",
                    address=fake.address(),
                    city=region,
                    region=region,
                    phone=fake.phone_number(),
                    email=f"{region.lower()}.main@bankingllm.uz",
                    manager_name=fake.name(),
                    branch_type="FULL_SERVICE",
                    operating_hours="09:00-18:00 Mon-Fri",
                    services="Deposits,Loans,Cards,Transfers,Currency Exchange",
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
                            address=fake.address(),
                            city=region,
                            region=region,
                            phone=fake.phone_number(),
                            email=f"{region.lower()}.{j}@bankingllm.uz",
                            manager_name=fake.name(),
                            branch_type=random.choice(["FULL_SERVICE", "DIGITAL"]),
                            operating_hours="09:00-18:00 Mon-Fri" if j % 2 == 0 else "09:00-20:00 Mon-Sat",
                            services="Deposits,Loans,Cards,Transfers",
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
                    maintenance_fee=sum_to_tiyin(5),
                    minimum_balance=sum_to_tiyin(100)
                ),
                Product(
                    product_code="CHK002",
                    product_name="Premium Checking",
                    product_category="ACCOUNT",
                    product_type="CHECKING",
                    description="Premium checking with higher limits and benefits",
                    base_interest_rate=1.0,
                    maintenance_fee=sum_to_tiyin(15),
                    minimum_balance=sum_to_tiyin(1000),
                    minimum_income=sum_to_tiyin(50000)
                ),
                # Savings Accounts
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
                # Business Accounts
                Product(
                    product_code="BUS001",
                    product_name="Business Checking",
                    product_category="ACCOUNT",
                    product_type="BUSINESS",
                    description="Business checking account for companies",
                    base_interest_rate=1.5,
                    maintenance_fee=sum_to_tiyin(25),
                    minimum_balance=sum_to_tiyin(500)
                ),
                # Loans
                Product(
                    product_code="LON001",
                    product_name="Personal Loan",
                    product_category="LOAN",
                    product_type="PERSONAL",
                    description="Personal loan for various needs",
                    base_interest_rate=12.0,
                    credit_limit=sum_to_tiyin(100000),
                    minimum_income=sum_to_tiyin(30000),
                    credit_score_required=650
                ),
                Product(
                    product_code="LON002",
                    product_name="Mortgage Loan",
                    product_category="LOAN",
                    product_type="MORTGAGE",
                    description="Home mortgage with competitive rates",
                    base_interest_rate=8.5,
                    credit_limit=sum_to_tiyin(5000000),
                    minimum_income=sum_to_tiyin(100000),
                    credit_score_required=700
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
                        "phone": fake.phone_number(),
                        "address": fake.address(),
                        "city": fake.city(),
                        "region": region,  # Based on occupation profile
                        "postal_code": fake.postcode(),
                        "occupation": occupation,
                        "income_level": income_level,
                        "risk_rating": random.choices(risk_ratings, weights=risk_weights)[0],
                        "status": random.choices(["ACTIVE", "INACTIVE", "SUSPENDED"], weights=status_weights)[0],
                        "kyc_status": random.choices(kyc_statuses, weights=kyc_weights)[0],
                        "country": "Uzbekistan",
                        "created_date": datetime.utcnow(),
                        "last_updated": datetime.utcnow(),
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
            currencies = ["UZS", "USD", "EUR"]

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
                    # Sophisticated currency selection based on occupation and profile
                    if client.occupation in ["Import/Export Trader", "IT Executive", "Bank Executive"]:
                        # International business professionals need foreign currencies
                        currency = random.choices(currencies, weights=[0.4, 0.4, 0.2])[0]
                    elif client.income_level == "ULTRA_HIGH":
                        currency = random.choices(currencies, weights=[0.5, 0.35, 0.15])[0]
                    elif client.income_level == "HIGH" and profile.get("tech_savvy", False):
                        currency = random.choices(currencies, weights=[0.6, 0.3, 0.1])[0]
                    elif client.occupation in ["Student", "Retired", "Agriculture Specialist"]:
                        # Domestic-focused occupations prefer local currency
                        currency = random.choices(currencies, weights=[0.9, 0.08, 0.02])[0]
                    else:
                        currency = random.choices(currencies, weights=[0.75, 0.2, 0.05])[0]

                    # Calculate realistic starting balance based on client profile (in tiyin)
                    base_balance = self._calculate_realistic_account_balance(client, account_type)

                    # Convert to other currencies (still stored in tiyin, but representing foreign currency cents)
                    if currency == "USD":
                        balance = base_balance // 125  # UZS tiyin to USD cents (12500 UZS tiyin = 100 USD cents)
                    elif currency == "EUR":
                        balance = base_balance // 135  # UZS tiyin to EUR cents (13500 UZS tiyin = 100 EUR cents)
                    else:
                        balance = base_balance  # UZS tiyin

                    # Realistic financial parameters (in tiyin)
                    overdraft_limit = 0
                    if account_type == "CHECKING" and balance > sum_to_tiyin(5000):  # 5000 сўм
                        overdraft_limit = min(int(balance * 0.15), sum_to_tiyin(30000))  # 15% of balance, max 30K сўм

                    daily_limit = min(int(balance * 0.3), sum_to_tiyin(150000))  # 30% daily limit, max 150K сўм

                    account = Account(
                        account_number=f"AC{account_counter:012d}",
                        client_id=client.id,
                        branch_id=random.choice(branch_ids),
                        account_type=account_type,
                        currency=currency,
                        balance=balance,
                        available_balance=int(balance * random.uniform(0.92, 0.99)),  # 1-8% holds
                        overdraft_limit=overdraft_limit,
                        daily_transaction_limit=daily_limit,
                        interest_rate=self._get_realistic_interest_rate(account_type, client.income_level),
                        monthly_fee=self._get_realistic_monthly_fee(account_type, balance),
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

            # Generate transactions with sophisticated risk patterns and intelligent algorithms
            logger.info("Generating transactions with intelligent risk scoring...")

            # Get accounts with client information for sophisticated transaction patterns
            accounts_with_clients = session.query(Account, Client).join(Client).all()
            account_client_map = {acc.id: (acc, client) for acc, client in accounts_with_clients}
            account_ids = list(account_client_map.keys())

            transaction_types = ["DEPOSIT", "WITHDRAWAL", "TRANSFER", "PAYMENT", "FEE"]
            transaction_subtypes = {
                "DEPOSIT": ["SALARY_DEPOSIT", "CASH_DEPOSIT", "CHECK_DEPOSIT", "TRANSFER_IN"],
                "WITHDRAWAL": ["ATM_WITHDRAWAL", "BRANCH_WITHDRAWAL", "ONLINE_WITHDRAWAL"],
                "TRANSFER": ["INTERNAL_TRANSFER", "EXTERNAL_TRANSFER", "WIRE_TRANSFER"],
                "PAYMENT": ["UTILITY_PAYMENT", "LOAN_PAYMENT", "MERCHANT_PAYMENT", "SUBSCRIPTION"],
                "FEE": ["MONTHLY_FEE", "OVERDRAFT_FEE", "ATM_FEE", "WIRE_FEE"]
            }
            channels = ["ATM", "ONLINE", "BRANCH", "MOBILE", "POS"]
            merchant_categories = ["Grocery", "Gas", "Restaurant", "Shopping", "Healthcare", "Utilities", "Entertainment"]

            # Generate chronologically ordered transactions with running balances
            transaction_counter = 1

            # Calculate transactions per account for even distribution
            transactions_per_account = settings.num_transactions // len(account_ids)
            remaining_transactions = settings.num_transactions % len(account_ids)

            # Track running balances for each account (in tiyin)
            account_running_balances = {}
            for account_id in account_ids:
                account, client = account_client_map[account_id]
                # Use account's initial balance as starting point (already in tiyin)
                if account.balance and account.balance > 0:
                    account_running_balances[account_id] = account.balance
                else:
                    # Generate random starting balance in tiyin (1K - 20K сўм)
                    account_running_balances[account_id] = sum_to_tiyin(random.uniform(1000, 20000))

            # Process transactions by account to maintain chronological order and running balances
            batch_size = 10000  # Smaller batches for better memory management
            all_transactions = []

            logger.info("Generating chronologically ordered transactions with running balances...")

            for account_idx, account_id in enumerate(account_ids):
                account, client = account_client_map[account_id]

                # Determine number of transactions for this account
                num_transactions_for_account = transactions_per_account
                if account_idx < remaining_transactions:
                    num_transactions_for_account += 1

                if num_transactions_for_account == 0:
                    continue

                # Generate chronologically ordered dates for this account
                account_transactions = []
                start_date = fake.date_time_between(start_date="-2y", end_date="-18m")
                end_date = fake.date_time_between(start_date="-1m", end_date="now")

                # Create sorted transaction dates
                transaction_dates = []
                for i in range(num_transactions_for_account):
                    tx_date = fake.date_time_between(start_date=start_date, end_date=end_date)
                    transaction_dates.append(tx_date)
                transaction_dates.sort()  # Ensure chronological order

                # Track running balance for this account (in tiyin)
                running_balance = account_running_balances[account_id]

                # Create recurring payment schedule for this client
                recurring_payments = self._create_recurring_schedule(client, account, transaction_dates)
                sequence_counter = 1

                for tx_date in transaction_dates:
                    # Store balance before transaction
                    balance_before = running_balance

                    # Intelligent transaction type selection based on client profile
                    profile = occupation_profiles.get(client.occupation, occupation_profiles["Engineer"])

                    # Sophisticated transaction type distribution based on occupation
                    if client.occupation in ["Business Owner", "Import/Export Trader", "Restaurant Owner"]:
                        # Business clients have more diverse transactions
                        type_weights = {"DEPOSIT": 0.35, "WITHDRAWAL": 0.20, "TRANSFER": 0.25, "PAYMENT": 0.15, "FEE": 0.05}
                    elif client.occupation in ["Student", "Retired"]:
                        # Simple transaction patterns
                        type_weights = {"DEPOSIT": 0.25, "WITHDRAWAL": 0.40, "TRANSFER": 0.15, "PAYMENT": 0.15, "FEE": 0.05}
                    else:
                        # Standard distribution
                        type_weights = {"DEPOSIT": 0.30, "WITHDRAWAL": 0.30, "TRANSFER": 0.20, "PAYMENT": 0.15, "FEE": 0.05}

                    # Smart transaction type selection with time-based logic
                    if self._is_business_hours(tx_date) and random.random() < 0.3:
                        # More likely to be salary/business transactions during business hours
                        type_weights = {"DEPOSIT": 0.4, "WITHDRAWAL": 0.2, "TRANSFER": 0.2, "PAYMENT": 0.15, "FEE": 0.05}
                    elif tx_date.hour >= 17 or tx_date.weekday() >= 5:  # After work or weekends
                        # More likely to be personal transactions
                        type_weights = {"WITHDRAWAL": 0.35, "PAYMENT": 0.25, "DEPOSIT": 0.2, "TRANSFER": 0.15, "FEE": 0.05}
                    else:
                        # Use the original distribution
                        pass  # Keep existing type_weights

                    transaction_type = random.choices(list(type_weights.keys()), weights=list(type_weights.values()))[0]
                    transaction_subtype = random.choice(transaction_subtypes[transaction_type])

                    # Risk-based transaction patterns
                    risk_factor = 0.0  # Start with base risk
                    risk_flags = []

                    # Smart channel selection based on client profile, transaction type, and time
                    channel = self._select_smart_channel(client, transaction_type, transaction_subtype, tx_date, profile)

                    # Use smart amount generation
                    amount = self._generate_smart_transaction_amount(client, transaction_type, transaction_subtype, tx_date, recurring_payments)

                    # Additional risk scoring based on patterns
                    if client.risk_rating == "HIGH" and random.random() < 0.3:
                        risk_factor += 0.4
                        risk_flags.append("HIGH_RISK_CLIENT")

                    # Unusual time patterns (late night transactions) - use the chronological date
                    if tx_date.hour < 6 or tx_date.hour > 22:
                        if channel in ["ATM", "ONLINE"] and abs(amount) > sum_to_tiyin(20000):  # >20K сўм
                            risk_factor += 0.15
                            risk_flags.append("UNUSUAL_HOURS")

                    # Calculate final risk score (0.0 to 1.0)
                    risk_score = min(risk_factor, 1.0)

                    # Intelligent flagging based on risk score and patterns
                    flagged_for_review = (
                        risk_score > 0.7 or  # High risk score
                        len(risk_flags) >= 2 or  # Multiple risk factors
                        "HIGH_RISK_CLIENT" in risk_flags  # High-risk client
                    )

                    # Realistic balance calculations with running balance
                    balance_after = running_balance + amount

                    # Prevent negative balances for most accounts (allow small overdrafts)
                    if balance_after < -account.overdraft_limit:
                        # Adjust transaction amount to stay within overdraft limit
                        amount = -account.overdraft_limit - running_balance
                        balance_after = running_balance + amount

                    # Smart risk calculation based on amount and patterns
                    if abs(amount) > sum_to_tiyin(50000):  # Large amounts >50K сўм
                        risk_factor += 0.3
                        risk_flags.append("LARGE_AMOUNT")

                    if transaction_subtype == "CASH_DEPOSIT" and abs(amount) > sum_to_tiyin(30000):  # >30K сўм cash
                        risk_factor += 0.2
                        risk_flags.append("LARGE_CASH")

                    # Create dict for bulk insert with intelligent risk scoring
                    transaction_data = {
                        "transaction_reference": f"TXN{transaction_counter:015d}",
                        "account_id": account_id,
                        "sequence_number": sequence_counter,
                        "transaction_type": transaction_type,
                        "transaction_subtype": transaction_subtype,
                        "amount": amount,
                        "currency": account.currency if hasattr(account, 'currency') else "UZS",
                        "fee_amount": sum_to_tiyin(random.choice([0, 50, 100, 250])),  # Fee in сўм converted to tiyin
                        "description": f"{transaction_subtype.replace('_', ' ').title()} - {client.occupation}",
                        "merchant_name": fake.company() if transaction_type == "PAYMENT" else None,
                        "merchant_category": random.choice(merchant_categories) if transaction_type == "PAYMENT" else None,
                        "location": self._generate_realistic_location(client, transaction_type, transaction_subtype),
                        "channel": channel,  # Use the intelligently selected channel
                        "balance_before": balance_before,
                        "balance_after": balance_after,
                        "available_balance_after": int(balance_after * 0.95),  # 5% hold for pending
                        "transaction_date": tx_date,
                        "processing_date": tx_date,
                        "value_date": tx_date,
                        "risk_score": round(risk_score, 3),  # Use calculated risk score
                        "flagged_for_review": flagged_for_review,  # Intelligent flagging
                        "created_by": "SYSTEM"
                    }
                    account_transactions.append(transaction_data)
                    transaction_counter += 1

                    # Update running balance and sequence for next transaction
                    running_balance = balance_after
                    sequence_counter += 1

                # Update account running balance
                account_running_balances[account_id] = running_balance

                # Add account transactions to the main list
                all_transactions.extend(account_transactions)

                # Process in batches to manage memory
                if len(all_transactions) >= batch_size:
                    session.bulk_insert_mappings(Transaction, all_transactions[:batch_size])
                    session.commit()
                    logger.info(f"Generated {transaction_counter-1} transactions ({((transaction_counter-1)/settings.num_transactions)*100:.1f}% complete)")
                    all_transactions = all_transactions[batch_size:]

            # Insert remaining transactions
            if all_transactions:
                session.bulk_insert_mappings(Transaction, all_transactions)
                session.commit()

            # Update account balances to match final transaction balances
            logger.info("Updating account balances to match final transaction balances...")
            for account_id, final_balance in account_running_balances.items():
                session.query(Account).filter(Account.id == account_id).update({
                    'balance': final_balance,  # Already in tiyin for exact precision
                    'available_balance': int(final_balance * 0.95)  # 5% hold for pending transactions
                })
            session.commit()

            logger.info("Mock data generation completed successfully with realistic chronological transactions!")

            # Print some statistics
            stats = self.get_database_stats(session)
            logger.info(f"Database statistics: {stats}")

        except Exception as e:
            logger.error(f"Error generating mock data: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def _create_recurring_schedule(self, client, account, transaction_dates):
        """Create smart recurring payment schedule for realistic banking patterns."""
        recurring_payments = {}

        # Monthly salary deposits (end of month)
        if client.income_level != "LOW" or client.occupation != "Retired":
            # Calculate consistent monthly salary (in сўм, convert to tiyin)
            if client.income_level == "ULTRA_HIGH":
                monthly_salary_sum = random.choice([150000, 180000, 200000, 220000, 250000])  # Round amounts
            elif client.income_level == "HIGH":
                monthly_salary_sum = random.choice([80000, 100000, 120000, 150000])
            elif client.income_level == "MEDIUM":
                monthly_salary_sum = random.choice([30000, 40000, 50000, 60000, 70000])
            else:
                monthly_salary_sum = random.choice([15000, 20000, 25000, 30000])

            monthly_salary = sum_to_tiyin(monthly_salary_sum)

            recurring_payments['SALARY'] = {
                'amount': monthly_salary,
                'day_range': (25, 30),  # End of month
                'type': 'DEPOSIT',
                'subtype': 'SALARY_DEPOSIT'
            }

        # Monthly bills (in сўм, convert to tiyin)
        utility_amounts_sum = [2000, 2500, 3000, 3500, 4000]  # Round utilities in сўм
        recurring_payments['UTILITIES'] = {
            'amount': sum_to_tiyin(random.choice(utility_amounts_sum)),
            'day_range': (1, 5),  # Beginning of month
            'type': 'PAYMENT',
            'subtype': 'UTILITY_PAYMENT'
        }

        # Rent payments (if not business owner)
        if client.occupation not in ["Business Owner", "Real Estate Agent"]:
            rent_amounts_sum = {
                "ULTRA_HIGH": [20000, 25000, 30000],
                "HIGH": [15000, 20000],
                "MEDIUM": [8000, 10000, 12000],
                "LOW": [5000, 6000, 8000]
            }
            recurring_payments['RENT'] = {
                'amount': sum_to_tiyin(random.choice(rent_amounts_sum[client.income_level])),
                'day_range': (1, 5),
                'type': 'PAYMENT',
                'subtype': 'RENT_PAYMENT'
            }

        # Loan payments (higher income clients have loans)
        if client.income_level in ["HIGH", "ULTRA_HIGH"] and random.random() < 0.6:
            loan_amounts_sum = {
                "HIGH": [10000, 12000, 15000],
                "ULTRA_HIGH": [20000, 25000, 30000]
            }
            recurring_payments['LOAN'] = {
                'amount': sum_to_tiyin(random.choice(loan_amounts_sum[client.income_level])),
                'day_range': (10, 15),
                'type': 'PAYMENT',
                'subtype': 'LOAN_PAYMENT'
            }

        return recurring_payments

    def _generate_smart_transaction_amount(self, client, transaction_type, transaction_subtype, tx_date, recurring_payments):
        """Generate realistic transaction amounts based on patterns and client behavior."""

        # Check if this should be a recurring payment
        for payment_type, schedule in recurring_payments.items():
            if (schedule['type'] == transaction_type and
                schedule['subtype'] == transaction_subtype and
                schedule['day_range'][0] <= tx_date.day <= schedule['day_range'][1]):

                # Add small seasonal variation for utilities
                if payment_type == 'UTILITIES':
                    if tx_date.month in [12, 1, 2, 6, 7, 8]:  # Winter/summer months
                        return int(schedule['amount'] * random.uniform(1.1, 1.3))  # 10-30% higher, keep tiyin precision
                    else:
                        return int(schedule['amount'] * random.uniform(0.9, 1.1))  # Normal variation, keep tiyin precision
                else:
                    return schedule['amount']  # Exact recurring amount

        # Smart amount patterns based on transaction type (all amounts in сўм, converted to tiyin)
        if transaction_type == "WITHDRAWAL":
            # ATM withdrawals are always round amounts
            if transaction_subtype == "ATM_WITHDRAWAL":
                round_amounts_sum = {
                    "Student": [500, 1000, 1500, 2000],
                    "Retired": [1000, 1500, 2000, 3000],
                    "Business Owner": [2000, 3000, 5000, 10000],
                    "IT Executive": [2000, 3000, 5000, 8000],
                    "Bank Executive": [3000, 5000, 8000, 10000]
                }
                amounts_sum = round_amounts_sum.get(client.occupation, [1000, 2000, 3000, 5000])
                return -sum_to_tiyin(random.choice(amounts_sum))
            else:
                # Other withdrawals vary by income but still prefer round amounts
                if client.income_level == "ULTRA_HIGH":
                    amounts_sum = [5000, 10000, 15000, 20000, 30000]
                elif client.income_level == "HIGH":
                    amounts_sum = [3000, 5000, 8000, 10000, 15000]
                elif client.income_level == "MEDIUM":
                    amounts_sum = [2000, 3000, 5000, 8000]
                else:
                    amounts_sum = [1000, 1500, 2000, 3000, 5000]
                return -sum_to_tiyin(random.choice(amounts_sum))

        elif transaction_type == "PAYMENT":
            if transaction_subtype == "MERCHANT_PAYMENT":
                # Grocery/shopping amounts by family size and income (in сўм)
                if client.occupation == "Student":
                    amounts_sum = [500, 750, 1000, 1250, 1500]
                    return -sum_to_tiyin(random.choice(amounts_sum))
                elif client.occupation == "Retired":
                    amounts_sum = [800, 1200, 1500, 2000]
                    return -sum_to_tiyin(random.choice(amounts_sum))
                else:
                    # Family shopping
                    base_amounts_sum = [1500, 2000, 2500, 3000, 4000, 5000]
                    if client.income_level in ["HIGH", "ULTRA_HIGH"]:
                        base_amounts_sum = [int(x * 1.5) for x in base_amounts_sum]
                    return -sum_to_tiyin(random.choice(base_amounts_sum))
            else:
                # Other payments vary by type (500 to 10K сўм)
                sum_amount = random.randint(500, 10000)
                return -sum_to_tiyin(sum_amount)

        elif transaction_type == "DEPOSIT":
            if transaction_subtype == "CASH_DEPOSIT":
                # Business owners have larger, more irregular cash deposits
                if client.occupation in ["Business Owner", "Restaurant Owner", "Transportation Business"]:
                    amounts_sum = [10000, 20000, 30000, 50000, 80000]
                    return sum_to_tiyin(random.choice(amounts_sum))
                else:
                    amounts_sum = [2000, 5000, 10000]
                    return sum_to_tiyin(random.choice(amounts_sum))
            else:
                # Regular deposits (1K to 20K сўм)
                sum_amount = random.randint(1000, 20000)
                return sum_to_tiyin(sum_amount)

        elif transaction_type == "TRANSFER":
            # Transfers are usually round amounts (in сўм)
            amounts_sum = [5000, 10000, 15000, 20000, 30000, 50000]
            amount = sum_to_tiyin(random.choice(amounts_sum))
            return amount if random.choice([True, False]) else -amount  # 50% in/out

        else:  # FEE
            # Banking fees are fixed amounts (in сўм)
            fee_amounts_sum = [100, 150, 250, 500, 750]
            return -sum_to_tiyin(random.choice(fee_amounts_sum))

    def _is_business_hours(self, tx_date):
        """Check if transaction is during business hours."""
        # Monday to Friday, 9 AM to 6 PM
        return (tx_date.weekday() < 5 and 9 <= tx_date.hour < 18)

    def _select_smart_channel(self, client, transaction_type, transaction_subtype, tx_date, profile):
        """Select realistic channel based on client behavior, transaction type, and timing."""

        # ATM withdrawals must use ATM
        if transaction_subtype == "ATM_WITHDRAWAL":
            return "ATM"

        # Branch transactions during business hours for certain types
        if (transaction_subtype in ["CASH_DEPOSIT", "BRANCH_WITHDRAWAL"] and
            self._is_business_hours(tx_date)):
            return "BRANCH"

        # Large amounts often require branch visits
        if transaction_type in ["DEPOSIT", "TRANSFER"] and random.random() < 0.3:
            if self._is_business_hours(tx_date):
                return "BRANCH"

        # Age-based preferences
        if client.occupation == "Retired":
            # Retirees prefer branch visits
            if self._is_business_hours(tx_date):
                return random.choices(["BRANCH", "ATM", "ONLINE"], weights=[0.6, 0.3, 0.1])[0]
            else:
                return random.choices(["ATM", "ONLINE"], weights=[0.7, 0.3])[0]

        elif client.occupation == "Student":
            # Students prefer digital channels
            return random.choices(["MOBILE", "ONLINE", "ATM", "POS"], weights=[0.4, 0.3, 0.2, 0.1])[0]

        # Tech-savvy professionals
        elif profile.get("tech_savvy", False):
            if tx_date.hour >= 18 or tx_date.weekday() >= 5:  # After work or weekends
                return random.choices(["MOBILE", "ONLINE", "ATM"], weights=[0.5, 0.3, 0.2])[0]
            else:
                return random.choices(["ONLINE", "MOBILE", "ATM", "BRANCH"], weights=[0.4, 0.3, 0.2, 0.1])[0]

        # Business owners
        elif client.occupation in ["Business Owner", "Restaurant Owner"]:
            if self._is_business_hours(tx_date):
                return random.choices(["BRANCH", "ONLINE", "MOBILE"], weights=[0.4, 0.3, 0.3])[0]
            else:
                return random.choices(["ONLINE", "MOBILE", "ATM"], weights=[0.4, 0.4, 0.2])[0]

        # Default distribution for everyone else
        else:
            return random.choices(["ONLINE", "ATM", "BRANCH", "MOBILE", "POS"], weights=[0.3, 0.25, 0.2, 0.2, 0.05])[0]

    def _generate_realistic_location(self, client, transaction_type, transaction_subtype):
        """Generate realistic locations based on client region and transaction type."""

        # Most transactions happen in client's region
        if random.random() < 0.85:  # 85% in home region
            region = client.region
        else:
            # Business travel or family visits to other regions
            other_regions = ["Tashkent", "Samarkand", "Bukhara", "Andijan", "Namangan", "Fergana", "Nukus"]
            other_regions = [r for r in other_regions if r != client.region]
            region = random.choice(other_regions)

        # Location details based on transaction type and client profile
        if transaction_subtype == "ATM_WITHDRAWAL":
            locations = [
                f"{region} ATM Center",
                f"{region} Shopping Mall ATM",
                f"{region} Bank Branch ATM",
                f"{region} Airport ATM" if region == "Tashkent" else f"{region} City Center ATM"
            ]
            return random.choice(locations)

        elif transaction_type == "PAYMENT":
            if client.occupation == "Student":
                locations = [
                    f"{region} University Campus",
                    f"{region} Student Cafeteria",
                    f"{region} Bookstore"
                ]
            elif client.occupation == "Business Owner":
                locations = [
                    f"{region} Business District",
                    f"{region} Commercial Center",
                    f"{region} Trade Center"
                ]
            elif client.occupation == "Retired":
                locations = [
                    f"{region} Medical Center",
                    f"{region} Pharmacy",
                    f"{region} Local Market"
                ]
            else:
                locations = [
                    f"{region} Shopping Center",
                    f"{region} Supermarket",
                    f"{region} Restaurant District"
                ]
            return random.choice(locations)

        elif transaction_subtype in ["CASH_DEPOSIT", "BRANCH_WITHDRAWAL"]:
            return f"{region} Bank Branch"

        elif transaction_type == "TRANSFER":
            if client.occupation == "Import/Export Trader":
                return f"{region} International Business Center"
            else:
                return f"{region} Online Banking"

        else:
            # Default location
            return f"{region}, {fake.city()}"

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

    def _get_realistic_monthly_fee(self, account_type: str, balance: int) -> int:
        """Calculate realistic monthly fees based on account type and balance (returns tiyin)."""
        # Fee structures (in сўм, will be converted to tiyin)
        fee_structures = {
            "CHECKING": {"base": 250, "waiver_balance": 10000},     # 250 сўм, waived if >10K сўм
            "SAVINGS": {"base": 150, "waiver_balance": 20000},      # 150 сўм, waived if >20K сўм
            "BUSINESS": {"base": 1000, "waiver_balance": 50000},    # 1000 сўм, waived if >50K сўм
            "CREDIT": {"base": 500, "waiver_balance": 0},           # 500 сўм, no waiver
            "DEPOSIT": {"base": 0, "waiver_balance": 0}             # No fees for deposits
        }

        structure = fee_structures.get(account_type, {"base": 200, "waiver_balance": 10000})

        # Convert waiver balance to tiyin for comparison
        waiver_balance_tiyin = sum_to_tiyin(structure["waiver_balance"]) if structure["waiver_balance"] > 0 else 0

        # Waive fee if balance is above threshold
        if balance >= waiver_balance_tiyin and waiver_balance_tiyin > 0:
            return 0

        return sum_to_tiyin(structure["base"])

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

    def execute_query(self, sql_query: str) -> List[dict]:
        """Execute a raw SQL query and return results."""
        session = self.get_session()
        try:
            result = session.execute(text(sql_query))
            columns = result.keys()
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
        finally:
            session.close()


# Global database manager instance
db_manager = DatabaseManager()