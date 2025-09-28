"""Expert Banking SQL Intelligence Service.

This module provides advanced natural language to SQL conversion specifically
designed for banking systems. It leverages Large Language Models to generate
precise SQL queries with comprehensive banking domain expertise.

Features:
- Multi-database aware SQL generation
- Banking-specific prompt engineering
- Comprehensive input validation and security
- Professional error handling and logging
"""

import re
import sqlite3
from typing import Dict, Optional, List, Any
from urllib.parse import urlparse

import requests
from loguru import logger

from .config import settings
from .exceptions import (
    LLMError,
    LLMServiceUnavailableError,
    SQLGenerationError,
    ValidationError,
    SecurityError
)


class LLMService:
    """Expert banking SQL generation service with comprehensive validation and error handling.

    This service converts natural language queries into precise SQL statements
    specifically optimized for banking and financial data analysis.

    Attributes:
        base_url: Ollama service base URL
        model: LLM model name to use for SQL generation
        max_retries: Maximum number of retry attempts for API calls
        timeout: Request timeout in seconds
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        """Initialize the LLM service with configuration validation.

        Args:
            base_url: Optional Ollama service URL override
            model: Optional model name override

        Raises:
            ValidationError: If configuration is invalid
        """
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.llm_model
        self.max_retries = getattr(settings, 'llm_max_retries', 3)
        self.timeout = getattr(settings, 'llm_timeout', 30)

        # Validate configuration
        self._validate_configuration()

        logger.info(f"Initialized LLM service: {self.model} at {self.base_url}")

    def _validate_configuration(self) -> None:
        """Validate LLM service configuration.

        Raises:
            ValidationError: If configuration is invalid
        """
        if not self.base_url:
            raise ValidationError("LLM base URL is required", field_name="base_url")

        if not self.model:
            raise ValidationError("LLM model name is required", field_name="model")

        # Validate URL format
        try:
            parsed = urlparse(self.base_url)
            if not parsed.scheme or not parsed.netloc:
                raise ValidationError(
                    f"Invalid LLM base URL format: {self.base_url}",
                    field_name="base_url",
                    invalid_value=self.base_url
                )
        except Exception as e:
            raise ValidationError(
                f"Invalid LLM base URL: {str(e)}",
                field_name="base_url",
                invalid_value=self.base_url
            )

    def _validate_user_query(self, user_query: str) -> None:
        """Validate user input query for security and format.

        Args:
            user_query: The user's natural language query

        Raises:
            ValidationError: If query format is invalid
            SecurityError: If query contains security violations
        """
        if not user_query or not isinstance(user_query, str):
            raise ValidationError(
                "Query must be a non-empty string",
                field_name="user_query",
                invalid_value=str(user_query)
            )

        # Check query length
        if len(user_query.strip()) < 3:
            raise ValidationError(
                "Query too short - minimum 3 characters required",
                field_name="user_query",
                invalid_value=user_query
            )

        if len(user_query) > 1000:
            raise ValidationError(
                "Query too long - maximum 1000 characters allowed",
                field_name="user_query",
                invalid_value=f"{user_query[:50]}..."
            )

        # Enhanced security checks for dangerous operations
        dangerous_patterns = [
            r'\b(DELETE|DROP|TRUNCATE|ALTER|CREATE|INSERT|UPDATE)\b',
            r'--',  # SQL comments
            r'/\*.*?\*/',  # Multi-line comments
            r';\s*(DELETE|DROP|TRUNCATE|ALTER|CREATE|INSERT|UPDATE)',  # Chained dangerous operations
            r'\bEXEC\b|\bEXECUTE\b',  # Stored procedure execution
            r'\bxp_\w+\b',  # Extended stored procedures
            r'\bsp_\w+\b',  # System stored procedures
        ]

        query_upper = user_query.upper()
        for pattern in dangerous_patterns:
            if re.search(pattern, query_upper, re.IGNORECASE):
                raise SecurityError(
                    "Query contains potentially dangerous SQL operations",
                    security_violation="dangerous_sql_operation",
                    blocked_content=user_query[:100]
                )

    def _validate_generated_sql(self, sql: str) -> None:
        """Validate generated SQL for syntax and security.

        Args:
            sql: Generated SQL query

        Raises:
            SQLGenerationError: If SQL is invalid or unsafe
        """
        if not sql or not isinstance(sql, str):
            raise SQLGenerationError(
                "Generated SQL is empty or invalid",
                user_query="",
                raw_response=sql
            )

        # Remove comments and normalize whitespace
        sql_clean = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
        sql_clean = ' '.join(sql_clean.split())

        # Must start with SELECT
        if not sql_clean.upper().strip().startswith('SELECT'):
            raise SQLGenerationError(
                "Generated SQL must start with SELECT statement",
                user_query="",
                raw_response=sql
            )

        # Basic SQL syntax validation using sqlite3 parser
        try:
            # Parse SQL to check basic syntax
            sqlite3.complete_statement(sql_clean)
        except Exception as e:
            raise SQLGenerationError(
                f"Generated SQL has syntax errors: {str(e)}",
                user_query="",
                raw_response=sql
            )

    def _create_expert_prompt(self, user_query: str) -> str:
        """Create expert banking prompt with real database context and examples.

        Args:
            user_query: Validated user query

        Returns:
            Comprehensive prompt for LLM with actual database context
        """
        # Get real database context for enhanced awareness
        try:
            from .database import db_manager
            db_context = db_manager.get_database_context_for_llm()
        except Exception as e:
            logger.warning(f"Could not load database context: {e}")
            db_context = {"branches": {"regions": ["Tashkent", "Samarkand", "Bukhara"]}}

        # Extract real data examples
        actual_branches = db_context.get("branches", {}).get("examples", [])[:5]
        actual_regions = db_context.get("branches", {}).get("regions", ["Tashkent", "Samarkand", "Bukhara"])
        client_counts = db_context.get("clients", {}).get("by_region", {})
        transaction_types = db_context.get("transactions", {}).get("types", ["DEPOSIT", "WITHDRAWAL"])

        return f"""BANKING SQL EXPERT - SQLite SYSTEM

CRITICAL: Only respond to banking/finance queries. For non-banking queries (greetings, weather, etc.) respond exactly: BANKING_REJECT
BANKING QUERIES: accounts, transactions, clients, balances, payments, loans, deposits, withdrawals, branches, KYC, AML, compliance
NON-BANKING: greetings, weather, sports, food, travel, general questions, programming, etc. → BANKING_REJECT

DATABASE: Single database containing all banking entities
SCHEMA: branches(id,branch_code,branch_name,city,region,branch_type,status,daily_cash_limit,max_transaction_amount,opened_date,created_date,created_by) | products(id,product_code,product_name,product_category,product_type,description,base_interest_rate,annual_fee,minimum_balance,risk_category,status,launch_date,created_date,created_by) | clients(id,client_number,name,birth_date,email,region,occupation,income_level,risk_rating,status,kyc_status,created_date,created_by) | accounts(id,account_number,client_id,branch_id,account_type,account_subtype,balance,available_balance,interest_rate,minimum_balance,status,open_date,created_date) | transactions(id,transaction_reference,account_id,transaction_type,transaction_subtype,amount,fee_amount,channel,balance_before,balance_after,risk_score,flagged_for_review,review_status,transaction_date,processing_date,status,created_by,authorized_by)

RELATIONSHIPS: clients.id=accounts.client_id=transactions.account_id | branches.id=accounts.branch_id

AMOUNTS: Stored in tiyin. Convert: amount/100.0 for som display
REGIONS: {", ".join(actual_regions)}
BRANCHES: {", ".join(actual_branches[:3]) if actual_branches else "Loading"}
CLIENTS: {" | ".join([f"{region}:{count:,}" for region, count in list(client_counts.items())[:3]])}

BANKING RULES:
- High risk: risk_score>0.7 OR flagged_for_review=1
- Large transactions: amount>2000000000 (20M UZS)
- Recent: transaction_date>=date('now','-30 days')
- Active: status='ACTIVE'
- KYC expired: kyc_status!='VALID'
- Suspicious patterns: Multiple large transactions same day

COMPLIANCE PATTERNS:
- AML threshold: 2000000000 tiyin (20M UZS)
- KYC required: transactions>500000000 tiyin (5M UZS)
- Risk scoring: 0.0-1.0 scale, >0.7 requires review
- Transaction velocity: >10 transactions/hour flagged

SQLite FUNCTIONS: date('now','-N days') | strftime('%Y-%m',date) | strftime('%H',datetime) | julianday() | COUNT() | SUM() | AVG() | GROUP BY | ORDER BY

MANDATORY SQL RULES:
- NEVER use SELECT * - Always specify exact column names
- Use explicit column selection: SELECT column1, column2, column3
- Example: SELECT c.name, c.region, a.balance NOT SELECT *

COMPREHENSIVE COLUMN SELECTION RULES:
- SELECT ALL relevant columns by explicit names - never use SELECT *
- Unless user specifies exact columns, include ALL meaningful fields
- For clients: ALWAYS include client_number, name, email, birth_date, region, occupation, income_level, risk_rating, status, kyc_status, created_date
- For accounts: ALWAYS include account_number, client_id, account_type, balance, available_balance, status, open_date
- For transactions: ALWAYS include transaction_reference, account_id, transaction_type, amount, transaction_date, status
- Add related contextual columns even when specific columns requested
- Prioritize banking compliance fields (kyc_status, risk_rating, status)

COMPLETE COLUMN EXAMPLES:
Client query: SELECT client_number, name, email, birth_date, region, occupation, income_level, risk_rating, status, kyc_status, created_date FROM clients WHERE region='Tashkent'
Account query: SELECT account_number, client_id, account_type, balance/100.0 as balance_som, available_balance/100.0 as available_som, status, open_date FROM accounts WHERE status='ACTIVE'

ESSENTIAL JOINS:
Regional client balances: FROM clients c JOIN accounts a ON c.id=a.client_id WHERE c.region='RegionName'
Transaction analysis: FROM transactions t JOIN accounts a ON t.account_id=a.id JOIN clients c ON a.client_id=c.id
Branch performance: FROM branches b LEFT JOIN accounts a ON b.id=a.branch_id

QUERY PATTERNS:
Balance inquiry: SELECT c.name, a.balance/100.0 as balance_som FROM clients c JOIN accounts a ON c.id=a.client_id WHERE c.region='Tashkent'
Transaction volume: SELECT c.region, SUM(t.amount)/100.0 as volume_som FROM transactions t JOIN accounts a ON t.account_id=a.id JOIN clients c ON a.client_id=c.id GROUP BY c.region
Risk analysis: SELECT COUNT(*) FROM transactions WHERE risk_score>0.7 OR flagged_for_review=1
KYC monitoring: SELECT COUNT(*) FROM clients WHERE kyc_status!='VALID'

USER QUERY: {user_query}

RESPONSE: SQLite SQL only. No explanations. No markdown formatting."""

    def _call_ollama(self, prompt: str) -> str:
        """Execute LLM API call with comprehensive error handling and retries.

        Args:
            prompt: The banking expert prompt to send to the LLM

        Returns:
            Raw response from the LLM

        Raises:
            LLMServiceUnavailableError: If service is unavailable after all retries
            LLMError: For other LLM-related errors
        """
        logger.info(f"Calling Ollama model: {self.model}")

        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{self.max_retries}")

                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.01,  # Very low for consistent SQL generation
                            "top_p": 0.95,       # Slightly higher for 14B model
                            "top_k": 50,         # Increased for better vocabulary
                            "num_predict": 800,  # Increased for complex banking queries
                            "num_ctx": 16384,    # Much larger context for 14B model
                            "repeat_penalty": 1.05,  # Lower penalty for 14B model
                            "stop": ["\n\n", "Query:", "Return", "BANKING_REJECT", "```"]
                        }
                    },
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        result = response_data.get("response", "").strip()

                        if not result:
                            logger.warning(f"Empty response from LLM on attempt {attempt + 1}")
                            continue

                        logger.info(f"✅ LLM response received ({len(result)} chars)")
                        logger.debug(f"Raw LLM response: '{result[:200]}...'")
                        return result

                    except ValueError as e:
                        logger.error(f"Invalid JSON response on attempt {attempt + 1}: {e}")
                        last_error = f"Invalid JSON response: {e}"
                        continue

                elif response.status_code == 404:
                    raise LLMError(
                        f"Model '{self.model}' not found. Please ensure the model is available.",
                        model_name=self.model
                    )
                elif response.status_code == 503:
                    logger.warning(f"Service unavailable on attempt {attempt + 1}, retrying...")
                    last_error = f"Service unavailable (HTTP {response.status_code})"
                    continue
                else:
                    logger.error(f"HTTP error {response.status_code} on attempt {attempt + 1}")
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    continue

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout after {self.timeout}s on attempt {attempt + 1}")
                last_error = f"Request timeout after {self.timeout}s"
                continue

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                last_error = f"Connection error: {str(e)}"
                continue

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                last_error = f"Request error: {str(e)}"
                continue

        # All attempts failed
        error_msg = f"LLM service unavailable after {self.max_retries} attempts"
        if last_error:
            error_msg += f". Last error: {last_error}"

        logger.error(error_msg)
        raise LLMServiceUnavailableError(
            error_msg,
            service_url=self.base_url,
            model_name=self.model
        )

    def _extract_sql(self, raw_response: str) -> str:
        """Ultra-clean SQL extraction with banking rejection detection."""
        if not raw_response:
            return ""

        # Check for banking rejection
        if "BANKING_REJECT" in raw_response.upper():
            return "BANKING_REJECT"

        # Clean response
        sql = raw_response.strip()

        # Remove code blocks if present
        if '```' in sql:
            sql = re.sub(r'```(?:sql)?\s*\n?([^`]+?)(?:\n?```|$)', r'\1', sql, flags=re.DOTALL)
            sql = sql.strip()

        # Clean whitespace and ensure semicolon
        sql = re.sub(r'\s+', ' ', sql.strip())
        if sql and not sql.endswith(';'):
            sql += ';'

        # Validate: must start with SELECT
        if sql.upper().startswith('SELECT'):
            return sql
        else:
            return ""

    def generate_sql(self, user_query: str) -> Dict[str, Any]:
        """Generate SQL from natural language with comprehensive validation and error handling.

        This is the main entry point for converting natural language queries into
        precise SQL statements optimized for banking data analysis.

        Args:
            user_query: Natural language query in English, Russian, or Uzbek

        Returns:
            Dictionary containing:
            - success: Boolean indicating if SQL was generated successfully
            - sql_query: Generated SQL string (if successful)
            - query_description: Original user query
            - error: Error message (if failed)
            - error_code: Structured error code for programmatic handling

        Raises:
            ValidationError: For invalid input
            SecurityError: For security violations
        """
        logger.info(f"🔍 Processing query: '{user_query[:100]}...'")

        try:
            # Step 1: Comprehensive input validation
            self._validate_user_query(user_query)

            # Step 2: Generate expert prompt
            prompt = self._create_expert_prompt(user_query)

            # Step 3: Call LLM with retry logic
            raw_response = self._call_ollama(prompt)

            # Step 4: Extract and clean SQL
            sql_query = self._extract_sql(raw_response)

            # Step 5: Handle special responses
            if sql_query == "BANKING_REJECT":
                logger.info("❌ Query rejected: not banking-related")
                return {
                    "success": False,
                    "sql_query": None,
                    "query_description": user_query,
                    "error": "Please ask questions related to banking, accounts, transactions, or financial data.",
                    "error_code": "NON_BANKING_QUERY"
                }

            # Step 6: Validate generated SQL
            if sql_query:
                self._validate_generated_sql(sql_query)
                logger.info(f"✅ Successfully generated SQL ({len(sql_query)} chars)")
                return {
                    "success": True,
                    "sql_query": sql_query,
                    "query_description": user_query
                }
            else:
                raise SQLGenerationError(
                    "Failed to extract valid SQL from LLM response",
                    user_query=user_query,
                    raw_response=raw_response
                )

        except (ValidationError, SecurityError) as e:
            # Re-raise validation and security errors for proper handling
            logger.error(f"❌ Validation/Security error: {e.message}")
            raise

        except SQLGenerationError as e:
            logger.error(f"❌ SQL generation error: {e.message}")
            return {
                "success": False,
                "sql_query": None,
                "query_description": user_query,
                "error": e.message,
                "error_code": e.error_code
            }

        except LLMServiceUnavailableError as e:
            logger.error(f"❌ LLM service unavailable: {e.message}")
            return {
                "success": False,
                "sql_query": None,
                "query_description": user_query,
                "error": "The AI service is currently unavailable. Please try again later.",
                "error_code": e.error_code
            }

        except LLMError as e:
            logger.error(f"❌ LLM error: {e.message}")
            return {
                "success": False,
                "sql_query": None,
                "query_description": user_query,
                "error": f"AI processing error: {e.message}",
                "error_code": e.error_code
            }

        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"❌ Unexpected error in SQL generation: {str(e)}", exc_info=True)
            return {
                "success": False,
                "sql_query": None,
                "query_description": user_query,
                "error": "An unexpected error occurred. Please try again or contact support.",
                "error_code": "UNEXPECTED_ERROR"
            }

    def suggest_sample_queries(self) -> list:
        """Expert banking queries demonstrating advanced SQL generation and domain knowledge."""
        return [
            # Regional Banking Intelligence
            "Show total transaction volume by region with growth trends",
            "Compare account balances across Tashkent and Samarkand clients",

            # Risk & Compliance Intelligence
            "Find clients with multiple large transactions in single day",
            "Show high-risk transactions requiring compliance review",

            # Financial Performance Analysis
            "Top 10 most profitable clients by transaction volume",
            "Monthly transaction patterns for business vs personal accounts",

            # Operational Intelligence
            "Average transaction amounts by channel and region",
            "Client distribution by income level and occupation"
        ]



# Global service instance
llm_service = LLMService()