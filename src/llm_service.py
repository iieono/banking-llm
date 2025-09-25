"""LLM service for natural language to SQL conversion."""

import json
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

import requests
from loguru import logger

from .config import settings


class LLMService:
    """Service for converting natural language to SQL queries using Ollama."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.llm_model
        self.schema_context = self._build_schema_context()

    def _build_schema_context(self) -> str:
        """Build database schema context for the LLM."""
        return """
DATABASE SCHEMA:

Table: clients
- id (INTEGER PRIMARY KEY): Unique client identifier
- name (TEXT): Client full name
- birth_date (DATETIME): Client date of birth
- region (TEXT): Client region (Tashkent, Samarkand, Bukhara, Andijan, Namangan, Fergana, Nukus)

Table: accounts
- id (INTEGER PRIMARY KEY): Unique account identifier
- client_id (INTEGER): Foreign key to clients table
- balance (REAL): Current account balance
- open_date (DATETIME): Date when account was opened

Table: transactions
- id (INTEGER PRIMARY KEY): Unique transaction identifier
- account_id (INTEGER): Foreign key to accounts table
- amount (REAL): Transaction amount (positive for credits, negative for debits)
- date (DATETIME): Transaction date
- type (TEXT): Transaction type (deposit, withdrawal, transfer_in, transfer_out, payment)

RELATIONSHIPS:
- clients.id → accounts.client_id (one-to-many)
- accounts.id → transactions.account_id (one-to-many)

REGIONS: Tashkent, Samarkand, Bukhara, Andijan, Namangan, Fergana, Nukus
"""

    def _create_prompt(self, user_query: str) -> str:
        """Create a structured prompt for SQL generation."""
        return f"""You are a SQL expert for a banking database. Convert the user's natural language request into a valid SQLite query.

{self.schema_context}

INSTRUCTIONS:
1. Generate ONLY the SQL query, no explanations or formatting
2. Use proper SQLite syntax
3. Use appropriate JOINs when accessing multiple tables
4. Use date functions like date() and datetime() for date operations
5. Always use table aliases for clarity
6. Return only SELECT statements (no INSERT, UPDATE, DELETE)
7. If the query involves dates, assume YYYY-MM-DD format
8. For regional queries, use exact region names from the list above
9. For aggregations, include appropriate GROUP BY clauses
10. Use LIMIT clause for top/bottom queries

USER REQUEST: {user_query}

SQL Query:"""

    def _validate_sql(self, sql_query: str) -> Tuple[bool, str]:
        """Validate the generated SQL query for safety and correctness."""
        # Remove any extra formatting
        sql_query = sql_query.strip().strip('`').strip()

        # Security checks
        dangerous_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER',
            'TRUNCATE', 'REPLACE', 'EXEC', 'EXECUTE', '--', '/*', '*/', ';'
        ]

        upper_query = sql_query.upper()
        for keyword in dangerous_keywords:
            if keyword in upper_query:
                return False, f"Query contains dangerous keyword: {keyword}"

        # Must start with SELECT
        if not upper_query.strip().startswith('SELECT'):
            return False, "Query must start with SELECT"

        # Basic syntax validation using sqlite3
        try:
            # Try to parse the query (this doesn't execute it)
            sqlite3.complete_statement(sql_query)
            return True, "Valid SQL"
        except Exception as e:
            return False, f"SQL syntax error: {str(e)}"

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API to generate response with retry logic."""
        for attempt in range(settings.llm_max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temperature for consistent SQL generation
                            "top_p": 0.9,
                            "num_predict": 200,  # Limit response length
                        }
                    },
                    timeout=settings.llm_timeout
                )

                if response.status_code == 200:
                    return response.json().get("response", "").strip()
                else:
                    logger.warning(f"Ollama API error (attempt {attempt + 1}): {response.status_code}")
                    if attempt == settings.llm_max_retries - 1:
                        logger.error(f"Final Ollama API error: {response.status_code} - {response.text}")
                    continue

            except requests.exceptions.Timeout:
                logger.warning(f"Ollama timeout (attempt {attempt + 1})")
                if attempt == settings.llm_max_retries - 1:
                    logger.error("Final Ollama timeout")
                continue
            except requests.exceptions.RequestException as e:
                logger.warning(f"Ollama request error (attempt {attempt + 1}): {e}")
                if attempt == settings.llm_max_retries - 1:
                    logger.error(f"Final Ollama error: {e}")
                continue

        return ""

    def generate_sql(self, user_query: str) -> Dict:
        """Generate SQL query from natural language input."""
        logger.info(f"Generating SQL for query: {user_query}")

        # Create prompt and call LLM
        prompt = self._create_prompt(user_query)
        raw_response = self._call_ollama(prompt)

        if not raw_response:
            return {
                "success": False,
                "error": "Failed to get response from LLM",
                "sql_query": None,
                "user_query": user_query
            }

        # Extract SQL from response (sometimes LLM adds extra text)
        sql_query = self._extract_sql(raw_response)

        # Validate the generated SQL
        is_valid, validation_message = self._validate_sql(sql_query)

        result = {
            "success": is_valid,
            "sql_query": sql_query if is_valid else None,
            "user_query": user_query,
            "raw_response": raw_response,
            "validation_message": validation_message
        }

        if not is_valid:
            result["error"] = validation_message
            logger.warning(f"Generated invalid SQL: {validation_message}")
        else:
            logger.info(f"Generated valid SQL: {sql_query}")

        return result

    def _extract_sql(self, response: str) -> str:
        """Extract SQL query from LLM response."""
        # Remove common prefixes and suffixes
        response = response.strip()

        # Try to find SQL in code blocks
        code_block_match = re.search(r'```(?:sql)?\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Try to find lines that look like SQL
        lines = response.split('\n')
        sql_lines = []

        for line in lines:
            line = line.strip()
            if line.upper().startswith('SELECT'):
                sql_lines.append(line)
            elif sql_lines and (line.upper().startswith(('FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT')) or line.endswith(';')):
                sql_lines.append(line)

        if sql_lines:
            return ' '.join(sql_lines).rstrip(';')

        # If no clear SQL found, return the whole response and let validation catch issues
        return response

    def get_query_explanation(self, sql_query: str) -> str:
        """Generate a human-readable explanation of the SQL query."""
        prompt = f"""Explain this SQL query in simple business terms for a banking context:

{sql_query}

Provide a brief, clear explanation of what this query does and what results it returns."""

        explanation = self._call_ollama(prompt)
        return explanation if explanation else "Unable to generate explanation"

    def suggest_sample_queries(self) -> List[str]:
        """Return a list of sample queries users can try."""
        return [
            "Show total transactions by region for 2024",
            "List top 10 clients by account balance",
            "Display monthly transaction trends for last 6 months",
            "Find accounts opened in Tashkent with balance above 50000",
            "Show transaction volume by type for each region",
            "Calculate average account balance by region",
            "List clients with negative transactions in the last month",
            "Show total deposits and withdrawals by month",
            "Find the most active accounts by transaction count",
            "Display regional distribution of clients"
        ]


# Global LLM service instance
llm_service = LLMService()