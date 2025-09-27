"""Ultra-Clean Banking SQL Intelligence Service.

Direct query→SQL conversion with 100% banking expertise.
Optimized for maximum speed and minimal complexity.
"""

import re
from typing import Dict

import requests
from loguru import logger

from .config import settings


class LLMService:
    """Ultra-clean banking SQL generation service."""

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.llm_model

    def _create_expert_prompt(self, user_query: str) -> str:
        """Expert banking prompt that generates perfect SQL with examples."""

        return f"""BANKING SQL EXPERT - RESPOND WITH PURE SQL ONLY

COMPLETE SCHEMA:
branches(id,branch_code,branch_name,city,region,branch_type,status,daily_cash_limit,max_transaction_amount,opened_date,created_date)
products(id,product_code,product_name,product_category,product_type,risk_category,status,interest_rate,annual_fee,minimum_balance,launch_date)
clients(id,client_number,name,birth_date,email,region,risk_rating,kyc_status,status,income_level,occupation,created_date,last_updated,created_by)
accounts(id,client_id,branch_id,account_number,account_type,account_subtype,balance,available_balance,currency,status,interest_rate,minimum_balance,open_date,close_date,last_transaction_date)
transactions(id,account_id,transaction_reference,transaction_type,transaction_subtype,amount,currency,fee_amount,channel,balance_before,balance_after,risk_score,flagged_for_review,review_status,transaction_date,processing_date,status,created_by,authorized_by)

RELATIONS: clients.id→accounts.client_id→transactions.account_id | branches.id→accounts.branch_id

RULES: amounts(÷100→UZS)|risk(HIGH/score>0.7)|recent(≥date('now','-30d'))|large(>20M tiyin)|active(=ACTIVE)|USD(×12000→UZS)|suspicious(flagged=1∨score>0.7)

EXPERT EXAMPLES:

High-risk clients:
SELECT client_number,name,region,risk_rating,kyc_status FROM clients WHERE risk_rating='HIGH' AND status='ACTIVE';

Large transactions with clients:
SELECT t.transaction_reference,t.amount/100.0 AS amount_uzs,c.name,a.account_number FROM transactions t JOIN accounts a ON t.account_id=a.id JOIN clients c ON a.client_id=c.id WHERE t.amount>20000000 AND t.status='COMPLETED';

KYC compliance issues:
SELECT client_number,name,kyc_status,created_date FROM clients WHERE kyc_status IN ('EXPIRED','PENDING') AND status='ACTIVE';

Suspicious transactions:
SELECT transaction_reference,transaction_type,amount/100.0 AS amount_uzs,risk_score,flagged_for_review FROM transactions WHERE flagged_for_review=1 OR risk_score>0.7;

Account balances with client info:
SELECT a.account_number,a.account_type,a.balance/100.0 AS balance_uzs,c.name,c.region FROM accounts a JOIN clients c ON a.client_id=c.id WHERE a.status='ACTIVE' ORDER BY a.balance DESC;

Recent transactions:
SELECT transaction_reference,transaction_type,amount/100.0 AS amount_uzs,channel,transaction_date FROM transactions WHERE transaction_date>=date('now','-7 days') AND status='COMPLETED' ORDER BY transaction_date DESC;

Regional analysis:
SELECT c.region,COUNT(DISTINCT c.id) as client_count,COUNT(DISTINCT a.id) as account_count,AVG(a.balance)/100.0 AS avg_balance_uzs FROM clients c JOIN accounts a ON c.id=a.client_id WHERE c.status='ACTIVE' GROUP BY c.region ORDER BY avg_balance_uzs DESC;

Transaction velocity monitoring:
SELECT c.client_number,c.name,COUNT(t.id) as daily_transactions,SUM(t.amount)/100.0 as daily_volume_uzs FROM clients c JOIN accounts a ON c.id=a.client_id JOIN transactions t ON a.id=t.account_id WHERE t.transaction_date>=date('now','-1 day') GROUP BY c.id HAVING COUNT(t.id)>10;

Multi-currency analysis:
SELECT a.currency,COUNT(*) as account_count,AVG(a.balance)/100.0 as avg_balance,SUM(CASE WHEN a.currency='USD' THEN a.balance*12000 ELSE a.balance END)/100.0 as uzs_equivalent FROM accounts a WHERE a.status='ACTIVE' GROUP BY a.currency ORDER BY uzs_equivalent DESC;

Branch performance:
SELECT b.branch_name,b.region,COUNT(DISTINCT a.id) as total_accounts,SUM(a.balance)/100.0 as total_deposits_uzs FROM branches b JOIN accounts a ON b.id=a.branch_id WHERE a.status='ACTIVE' GROUP BY b.id ORDER BY total_deposits_uzs DESC;

Client ranking by region:
SELECT client_number,name,balance/100.0 as balance_uzs,RANK() OVER (PARTITION BY region ORDER BY balance DESC) as regional_rank FROM clients c JOIN accounts a ON c.id=a.client_id WHERE c.status='ACTIVE' AND a.account_type='SAVINGS';

Risk correlation analysis:
SELECT c.risk_rating,AVG(t.risk_score) as avg_transaction_risk,COUNT(CASE WHEN t.flagged_for_review=1 THEN 1 END) as flagged_count FROM clients c JOIN accounts a ON c.id=a.client_id JOIN transactions t ON a.id=t.account_id WHERE t.transaction_date>=date('now','-30 days') GROUP BY c.risk_rating;

Compliance audit with violations:
SELECT c.client_number,c.name,a.account_type,COUNT(t.id) as high_risk_transactions FROM clients c JOIN accounts a ON c.id=a.client_id JOIN transactions t ON a.id=t.account_id WHERE c.kyc_status='EXPIRED' AND t.risk_score>0.7 GROUP BY c.id HAVING COUNT(t.id)>5;

Transaction channel analysis:
SELECT channel,COUNT(*) as transaction_count,SUM(amount)/100.0 AS total_volume_uzs,AVG(amount)/100.0 as avg_amount_uzs FROM transactions WHERE transaction_date>=date('now','-30 days') GROUP BY channel ORDER BY transaction_count DESC;

Income level demographics:
SELECT income_level,COUNT(*) as client_count,AVG(a.balance)/100.0 as avg_balance_uzs FROM clients c JOIN accounts a ON c.id=a.client_id WHERE c.status='ACTIVE' GROUP BY income_level ORDER BY avg_balance_uzs DESC;

MULTILINGUAL TERMS:
EN/RU/UZ: client/клиент/mijoz→clients | transaction/транзакция/tranzaksiya→transactions | account/счет/hisob→accounts | branch/филиал/filial→branches | risk/риск/xavf→risk_rating | high/высокий/yuqori→HIGH | balance/баланс/balans→balance | region/регион/hudud→region | suspicious/подозрительный/shubhali→flagged_for_review=1 | large/большой/katta→amount>20000000 | expired/просроченный/muddati tugagan→kyc_status='EXPIRED' | active/активный/faol→status='ACTIVE' | recent/недавний/yaqinda→>=date('now','-30d') | daily/ежедневно/kunlik→date('now','-1d') | currency/валюта/valyuta→currency

Query: {user_query}

RESPOND WITH ONLY THE SQL QUERY (no explanations, no markdown):"""

    def _call_ollama(self, prompt: str) -> str:
        """Optimized Ollama API call for banking SQL generation."""
        logger.info(f"Calling Ollama: {self.model}")

        for attempt in range(settings.llm_max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.01,
                            "top_p": 0.9,
                            "top_k": 40,
                            "num_predict": 250,
                            "num_ctx": 4096,
                            "repeat_penalty": 1.1,
                            "stop": ["\n\n", "Query:", "Return"]
                        }
                    },
                    timeout=settings.llm_timeout
                )

                if response.status_code == 200:
                    result = response.json().get("response", "").strip()
                    logger.info(f"Raw LLM response: '{result}'")
                    return result
                else:
                    logger.error(f"Ollama error (attempt {attempt + 1}): {response.status_code}")
                    continue

            except requests.exceptions.Timeout:
                logger.error(f"Timeout after {settings.llm_timeout}s (attempt {attempt + 1})")
                continue
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                continue

        logger.error("All Ollama attempts failed")
        return ""

    def _extract_sql(self, raw_response: str) -> str:
        """Ultra-clean SQL extraction - one workflow only."""
        if not raw_response:
            return ""

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

    def generate_sql(self, user_query: str) -> Dict:
        """
        Ultra-clean SQL generation: query → SQL
        Direct workflow with minimal validation.
        """

        # Basic security check
        dangerous_words = ['DELETE', 'UPDATE', 'INSERT', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
        query_upper = user_query.upper()
        for word in dangerous_words:
            if word in query_upper:
                logger.warning(f"Blocked dangerous operation: {word}")
                return {
                    "success": False,
                    "sql_query": None,
                    "error": f"Only SELECT queries allowed. Blocked: {word}"
                }

        try:
            # Direct generation workflow
            prompt = self._create_expert_prompt(user_query)
            raw_response = self._call_ollama(prompt)

            if raw_response:
                sql_query = self._extract_sql(raw_response)

                if sql_query and sql_query.upper().strip().startswith('SELECT'):
                    logger.info(f"Successfully generated SQL: {sql_query[:100]}...")
                    return {
                        "success": True,
                        "sql_query": sql_query,
                        "query_description": user_query
                    }
                else:
                    logger.warning("Failed to extract valid SQL")
            else:
                logger.warning("No response from LLM")

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")

        # No fallback - pure workflow as requested
        return {
            "success": False,
            "sql_query": None,
            "error": "Failed to generate SQL query"
        }


# Global service instance
llm_service = LLMService()