"""Database models and operations for Bank AI LLM system."""

import random
from datetime import datetime, timedelta
from typing import List, Optional

from faker import Faker
from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    func,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session

from .config import settings

Base = declarative_base()


class Client(Base):
    """Client model representing bank customers."""

    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    birth_date = Column(DateTime, nullable=False)
    region = Column(String(50), nullable=False, index=True)

    # Relationships
    accounts = relationship("Account", back_populates="client", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Client(id={self.id}, name='{self.name}', region='{self.region}')>"


class Account(Base):
    """Account model representing bank accounts."""

    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False, index=True)
    balance = Column(Float, nullable=False, default=0.0, index=True)
    open_date = Column(DateTime, nullable=False, index=True)

    # Relationships
    client = relationship("Client", back_populates="accounts")
    transactions = relationship("Transaction", back_populates="account", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Account(id={self.id}, client_id={self.client_id}, balance={self.balance})>"


class Transaction(Base):
    """Transaction model representing bank transactions."""

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    type = Column(String(20), nullable=False, index=True)

    # Relationships
    account = relationship("Account", back_populates="transactions")

    def __repr__(self):
        return f"<Transaction(id={self.id}, account_id={self.account_id}, amount={self.amount}, type='{self.type}')>"


class DatabaseManager:
    """Database manager for creating connections and managing data."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database_url

        # Optimized engine configuration
        self.engine = create_engine(
            self.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=10,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=settings.log_level == "DEBUG"  # SQL logging in debug mode
        )

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            expire_on_commit=False  # Performance optimization
        )

    def create_tables(self):
        """Create all database tables."""
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")

    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()

    def generate_mock_data(self):
        """Generate mock banking data with realistic patterns."""
        logger.info(f"Generating mock data: {settings.num_clients} clients, {settings.num_transactions} transactions")

        fake = Faker()
        fake.seed_instance(42)  # For reproducible data
        random.seed(42)

        session = self.get_session()

        try:
            # Clear existing data
            session.execute(text("DELETE FROM transactions"))
            session.execute(text("DELETE FROM accounts"))
            session.execute(text("DELETE FROM clients"))
            session.commit()

            # Generate clients
            regions = ["Tashkent", "Samarkand", "Bukhara", "Andijan", "Namangan", "Fergana", "Nukus"]
            clients = []

            logger.info("Generating clients...")
            for i in range(settings.num_clients):
                client = Client(
                    name=fake.name(),
                    birth_date=fake.date_of_birth(minimum_age=18, maximum_age=80),
                    region=random.choice(regions)
                )
                clients.append(client)

                if (i + 1) % 1000 == 0:
                    session.add_all(clients)
                    session.commit()
                    clients = []
                    logger.info(f"Generated {i + 1} clients")

            if clients:
                session.add_all(clients)
                session.commit()

            logger.info("Generating accounts...")
            # Generate accounts
            client_ids = [c.id for c in session.query(Client).all()]
            accounts = []
            account_id_counter = 1

            for client_id in client_ids:
                num_accounts = random.randint(*settings.num_accounts_per_client_range)
                for _ in range(num_accounts):
                    account = Account(
                        id=account_id_counter,
                        client_id=client_id,
                        balance=round(random.uniform(100, 100000), 2),
                        open_date=fake.date_between(start_date="-5y", end_date="today")
                    )
                    accounts.append(account)
                    account_id_counter += 1

                    if len(accounts) >= 1000:
                        session.add_all(accounts)
                        session.commit()
                        accounts = []

            if accounts:
                session.add_all(accounts)
                session.commit()

            # Generate transactions
            logger.info("Generating transactions...")
            account_ids = [a.id for a in session.query(Account).all()]
            transaction_types = ["deposit", "withdrawal", "transfer_in", "transfer_out", "payment"]
            transactions = []

            for i in range(settings.num_transactions):
                account_id = random.choice(account_ids)
                transaction_type = random.choice(transaction_types)

                # Generate realistic amounts based on transaction type
                if transaction_type in ["deposit", "transfer_in"]:
                    amount = round(random.uniform(10, 5000), 2)
                elif transaction_type in ["withdrawal", "transfer_out", "payment"]:
                    amount = round(random.uniform(10, 2000), 2) * -1
                else:
                    amount = round(random.uniform(10, 5000), 2)

                transaction = Transaction(
                    account_id=account_id,
                    amount=amount,
                    date=fake.date_time_between(start_date="-2y", end_date="now"),
                    type=transaction_type
                )
                transactions.append(transaction)

                if (i + 1) % 10000 == 0:
                    session.add_all(transactions)
                    session.commit()
                    transactions = []
                    logger.info(f"Generated {i + 1} transactions")

            if transactions:
                session.add_all(transactions)
                session.commit()

            logger.info("Mock data generation completed successfully")

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
                "regions": [r[0] for r in session.query(Client.region.distinct())],
                "date_range": {
                    "first_transaction": session.query(func.min(Transaction.date)).scalar(),
                    "last_transaction": session.query(func.max(Transaction.date)).scalar(),
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