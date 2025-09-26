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

Table: branches
- id (INTEGER PRIMARY KEY): Unique branch identifier
- branch_code (TEXT UNIQUE): Branch code (e.g., TAS001, SAM002)
- branch_name (TEXT): Branch name
- address (TEXT): Branch address
- city (TEXT): Branch city
- region (TEXT): Branch region
- phone (TEXT): Branch phone number
- email (TEXT): Branch email
- manager_name (TEXT): Branch manager name
- branch_type (TEXT): FULL_SERVICE, ATM_ONLY, DIGITAL, CORPORATE
- status (TEXT): ACTIVE, INACTIVE, CLOSED
- operating_hours (TEXT): Branch operating hours

Table: products
- id (INTEGER PRIMARY KEY): Unique product identifier
- product_code (TEXT UNIQUE): Product code (e.g., CHK001, SAV001)
- product_name (TEXT): Product name
- product_category (TEXT): ACCOUNT, LOAN, CARD, INVESTMENT, INSURANCE
- product_type (TEXT): CHECKING, SAVINGS, MORTGAGE, CREDIT_CARD, etc.
- description (TEXT): Product description
- currency (TEXT): UZS, USD, EUR
- base_interest_rate (REAL): Base interest rate
- annual_fee (REAL): Annual fee
- minimum_balance (REAL): Minimum balance required
- credit_limit (REAL): Credit limit for loans/cards
- status (TEXT): ACTIVE, DISCONTINUED, SUSPENDED

Table: clients
- id (INTEGER PRIMARY KEY): Unique client identifier
- client_number (TEXT UNIQUE): Client number (e.g., CL00000001)
- name (TEXT): Client full name
- birth_date (DATETIME): Client date of birth
- email (TEXT): Client email
- phone (TEXT): Client phone number
- address (TEXT): Client address
- city (TEXT): Client city
- region (TEXT): Client region (Tashkent, Samarkand, Bukhara, Andijan, Namangan, Fergana, Nukus)
- country (TEXT): Client country (default: Uzbekistan)
- postal_code (TEXT): Postal code
- occupation (TEXT): Client occupation
- income_level (TEXT): LOW, MEDIUM, HIGH, ULTRA_HIGH
- risk_rating (TEXT): LOW, MEDIUM, HIGH
- status (TEXT): ACTIVE, INACTIVE, SUSPENDED, CLOSED
- kyc_status (TEXT): PENDING, VERIFIED, EXPIRED
- created_date (DATETIME): Client creation date
- last_updated (DATETIME): Last update timestamp

Table: accounts
- id (INTEGER PRIMARY KEY): Unique account identifier
- account_number (TEXT UNIQUE): Account number (e.g., AC000000000001)
- client_id (INTEGER): Foreign key to clients table
- branch_id (INTEGER): Foreign key to branches table
- account_type (TEXT): CHECKING, SAVINGS, BUSINESS, INVESTMENT, LOAN
- account_subtype (TEXT): Specific account subtype
- currency (TEXT): UZS, USD, EUR
- balance (REAL): Current account balance
- available_balance (REAL): Available balance (after holds)
- overdraft_limit (REAL): Overdraft limit
- daily_transaction_limit (REAL): Daily transaction limit
- interest_rate (REAL): Account interest rate
- monthly_fee (REAL): Monthly maintenance fee
- minimum_balance (REAL): Minimum balance requirement
- status (TEXT): ACTIVE, INACTIVE, CLOSED, FROZEN
- open_date (DATETIME): Account opening date
- close_date (DATETIME): Account closing date
- last_transaction_date (DATETIME): Last transaction date

Table: transactions
- id (INTEGER PRIMARY KEY): Unique transaction identifier
- transaction_reference (TEXT UNIQUE): Transaction reference (e.g., TXN000000000000001)
- account_id (INTEGER): Foreign key to accounts table
- transaction_type (TEXT): DEPOSIT, WITHDRAWAL, TRANSFER, PAYMENT, FEE
- transaction_subtype (TEXT): Specific transaction subtype
- amount (REAL): Transaction amount
- currency (TEXT): Transaction currency
- original_amount (REAL): Original amount for currency conversions
- original_currency (TEXT): Original currency
- exchange_rate (REAL): Exchange rate used
- fee_amount (REAL): Fee charged for transaction
- description (TEXT): Transaction description
- merchant_name (TEXT): Merchant name for payments
- merchant_category (TEXT): Merchant category
- location (TEXT): Transaction location
- channel (TEXT): ATM, ONLINE, BRANCH, MOBILE, POS
- device_id (TEXT): Device identifier
- ip_address (TEXT): IP address for online transactions
- balance_before (REAL): Account balance before transaction
- balance_after (REAL): Account balance after transaction
- available_balance_after (REAL): Available balance after transaction
- status (TEXT): PENDING, COMPLETED, FAILED, CANCELLED
- related_account_id (INTEGER): Related account for transfers
- related_transaction_id (INTEGER): Related transaction ID
- risk_score (REAL): Risk score (0.0-1.0)
- flagged_for_review (BOOLEAN): Whether flagged for review
- review_status (TEXT): CLEARED, SUSPICIOUS, BLOCKED
- transaction_date (DATETIME): Transaction date
- processing_date (DATETIME): Processing date
- value_date (DATETIME): Value date

RELATIONSHIPS:
- branches.id → accounts.branch_id (one-to-many)
- clients.id → accounts.client_id (one-to-many)
- accounts.id → transactions.account_id (one-to-many)

REGIONS: Tashkent, Samarkand, Bukhara, Andijan, Namangan, Fergana, Nukus

ACCOUNT TYPES: CHECKING, SAVINGS, BUSINESS, INVESTMENT, LOAN

TRANSACTION TYPES: DEPOSIT, WITHDRAWAL, TRANSFER, PAYMENT, FEE

CHANNELS: ATM, ONLINE, BRANCH, MOBILE, POS

CURRENCIES: UZS (Uzbek Som / so'm), USD (US Dollar), EUR (Euro)

BANKING TERMINOLOGY (English/Russian/Uzbek):
- KYC (Know Your Customer): Client verification status (PENDING, VERIFIED, EXPIRED)
  RU: КУК (Знай Клиента): статус верификации (ОЖИДАНИЕ, ВЕРИФИЦИРОВАН, ИСТЁК)
  UZ: Mijozni bilish: tekshirish holati (KUTILMOQDA, TASDIQLANGAN, MUDDATI TUGAGAN)

- AML (Anti-Money Laundering): Compliance monitoring for suspicious activities
  RU: ОПД (Противодействие отмыванию денег): мониторинг подозрительных операций
  UZ: Pul yuvishga qarshi kurash: shubhali operatsiyalarni monitoring

- Account Balance: Current account amount (stored in tiyin, displayed in UZS/so'm)
  RU: Баланс счета: текущая сумма счета (хранится в тийинах, показывается в сумах)
  UZ: Hisob balansi: joriy hisob miqdori (tiyin saqlanadi, so'm ko'rsatiladi)

- Risk Rating: Client risk assessment (LOW, MEDIUM, HIGH)
  RU: Рисковая оценка: оценка риска клиента (НИЗКИЙ, СРЕДНИЙ, ВЫСОКИЙ)
  UZ: Risk baholash: mijozning xavf darajasi (PAST, O'RTA, YUQORI)

- Income Level: Client income category (LOW, MEDIUM, HIGH, ULTRA_HIGH)
  RU: Уровень дохода: категория дохода клиента (НИЗКИЙ, СРЕДНИЙ, ВЫСОКИЙ, СВЕРХВЫСОКИЙ)
  UZ: Daromad darajasi: mijozning daromad toifasi (PAST, O'RTA, YUQORI, JUDA_YUQORI)

- Transaction Channel: How transaction was initiated
  RU: Канал операции: способ инициации операции
  UZ: Operatsiya kanali: operatsiya qanday boshlangan
  • ATM: Bankomat / банкомат / bankomat
  • ONLINE: Онлайн / onlayn
  • BRANCH: Филиал / filial
  • MOBILE: Мобильный / mobil
  • POS: Терминал / terminal

- Overdraft: Negative account balance allowed up to a limit
  RU: Овердрафт: разрешенный отрицательный баланс до лимита
  UZ: Overdraft: limitgacha ruxsat etilgan manfiy balans

- Available Balance: Balance minus holds and pending transactions
  RU: Доступный баланс: баланс минус блокировки и ожидающие операции
  UZ: Mavjud balans: balans minus bloklar va kutilayotgan operatsiyalar

AMOUNT CONVERSION (Tiyin ↔ Currency):
- 1 so'm = 100 tiyin (Uzbek)
- 1 сум = 100 тийин (Russian)
- 1 UZS = 100 tiyin (English/ISO)
- Database stores amounts in tiyin for precision
- Display formats: 2000.30 UZS / 2000.30 so'm / 2000.30 сум
- Large amounts: >50,000 UZS (>5,000,000 tiyin)
- RU: 1 сум = 100 тийин, крупные суммы >50,000 сум
- UZ: 1 so'm = 100 tiyin, katta summalar >50,000 so'm

COMMON BANKING QUERY PATTERNS:

1. EXECUTIVE DASHBOARD & KPI ANALYSIS:
- "Show total assets under management by region with client count and average account balance"
  SELECT b.region, COUNT(DISTINCT c.id) as client_count, COUNT(a.id) as account_count,
         ROUND(SUM(a.balance), 2) as total_aum, ROUND(AVG(a.balance), 2) as avg_balance
  FROM branches b JOIN accounts a ON b.id = a.branch_id JOIN clients c ON a.client_id = c.id
  WHERE a.status = 'ACTIVE' GROUP BY b.region ORDER BY total_aum DESC;
- "Analyze transaction velocity by client occupation and identify top performing business segments"
  SELECT c.occupation, COUNT(t.id) as txn_count, ROUND(SUM(ABS(t.amount)), 2) as total_volume,
         ROUND(AVG(ABS(t.amount)), 2) as avg_amount, COUNT(DISTINCT c.id) as client_count
  FROM transactions t JOIN accounts a ON t.account_id = a.id JOIN clients c ON a.client_id = c.id
  WHERE t.transaction_date >= date('now', '-90 days') GROUP BY c.occupation ORDER BY total_volume DESC LIMIT 10;

2. RISK MANAGEMENT & ANTI-MONEY LAUNDERING:
- "Identify high-risk transactions with multiple risk flags and client occupation analysis"
  SELECT t.transaction_reference, t.amount, t.risk_score, c.name, c.occupation, c.income_level,
         t.channel, t.transaction_type, t.description
  FROM transactions t JOIN accounts a ON t.account_id = a.id JOIN clients c ON a.client_id = c.id
  WHERE t.flagged_for_review = 1 AND t.risk_score > 0.7 ORDER BY t.risk_score DESC, t.amount DESC LIMIT 25;
- "Show suspicious cash deposit patterns exceeding 5M UZS from business owners and restaurant operators"
  SELECT c.name, c.occupation, t.transaction_reference, t.amount, t.transaction_date, t.risk_score,
         COUNT(*) OVER (PARTITION BY c.id) as total_cash_deposits
  FROM transactions t JOIN accounts a ON t.account_id = a.id JOIN clients c ON a.client_id = c.id
  WHERE t.transaction_type = 'DEPOSIT' AND t.transaction_subtype = 'CASH_DEPOSIT'
        AND t.amount > 5000000 AND c.occupation IN ('Business Owner', 'Restaurant Owner')
  ORDER BY t.amount DESC, t.risk_score DESC;

3. CLIENT BEHAVIOR & SEGMENTATION ANALYTICS:
- "Analyze banking channel preferences by client age group and tech-savvy occupations"
  SELECT CASE WHEN (julianday('now') - julianday(c.birth_date))/365 < 35 THEN 'Young (18-35)'
              WHEN (julianday('now') - julianday(c.birth_date))/365 < 50 THEN 'Middle (35-50)'
              ELSE 'Senior (50+)' END as age_group,
         c.occupation, t.channel, COUNT(*) as usage_count
  FROM transactions t JOIN accounts a ON t.account_id = a.id JOIN clients c ON a.client_id = c.id
  WHERE c.occupation IN ('IT Executive', 'Bank Executive', 'Business Owner', 'Doctor/Surgeon', 'Student')
  GROUP BY age_group, c.occupation, t.channel ORDER BY c.occupation, age_group, usage_count DESC;
- "Identify VIP clients (ultra-high income) with multi-currency accounts and their preferred transaction types"
  SELECT c.client_number, c.name, c.occupation, COUNT(DISTINCT a.currency) as currencies,
         ROUND(SUM(a.balance), 2) as total_balance, t.transaction_type, COUNT(t.id) as txn_count
  FROM clients c JOIN accounts a ON c.id = a.client_id JOIN transactions t ON a.id = t.account_id
  WHERE c.income_level = 'ULTRA_HIGH' AND a.status = 'ACTIVE'
  GROUP BY c.id, t.transaction_type HAVING currencies > 1 ORDER BY total_balance DESC, txn_count DESC;

4. BUSINESS INTELLIGENCE & PERFORMANCE METRICS:
- "Calculate customer lifetime value by analyzing transaction frequency and average amounts per occupation"
  SELECT c.occupation, COUNT(DISTINCT c.id) as client_count,
         ROUND(AVG(txn_count), 2) as avg_txns_per_client, ROUND(AVG(total_volume), 2) as avg_volume_per_client
  FROM clients c JOIN (SELECT a.client_id, COUNT(t.id) as txn_count, SUM(ABS(t.amount)) as total_volume
                       FROM transactions t JOIN accounts a ON t.account_id = a.id GROUP BY a.client_id) tm
  ON c.id = tm.client_id GROUP BY c.occupation ORDER BY avg_volume_per_client DESC;
- "Show client acquisition trends by region with occupation distribution and income levels"
  SELECT b.region, c.income_level, c.occupation, COUNT(*) as client_count,
         ROUND(AVG(julianday('now') - julianday(c.created_date)), 0) as avg_days_since_onboarding
  FROM clients c JOIN accounts a ON c.id = a.client_id JOIN branches b ON a.branch_id = b.id
  WHERE c.created_date >= date('now', '-12 months') GROUP BY b.region, c.income_level, c.occupation
  ORDER BY b.region, client_count DESC;

5. COMPLIANCE & REGULATORY REPORTING:
- "Detect potential structuring patterns with multiple transactions just below reporting thresholds"
  SELECT c.client_number, c.name, c.occupation, COUNT(*) as near_threshold_txns, ROUND(SUM(t.amount), 2) as total_amount,
         MIN(t.transaction_date) as first_txn, MAX(t.transaction_date) as last_txn
  FROM transactions t JOIN accounts a ON t.account_id = a.id JOIN clients c ON a.client_id = c.id
  WHERE t.transaction_type = 'DEPOSIT' AND t.amount BETWEEN 4500000 AND 4999999
  GROUP BY c.id HAVING near_threshold_txns >= 3 ORDER BY near_threshold_txns DESC, total_amount DESC;
- "Find clients with inconsistent transaction patterns relative to their declared occupation and income level"
  SELECT c.client_number, c.name, c.occupation, c.income_level, ROUND(AVG(ABS(t.amount)), 2) as avg_txn_amount,
         MAX(ABS(t.amount)) as max_txn_amount, COUNT(CASE WHEN ABS(t.amount) > 10000000 THEN 1 END) as large_txns
  FROM clients c JOIN accounts a ON c.id = a.client_id JOIN transactions t ON a.id = t.account_id
  WHERE c.occupation IN ('Teacher/Professor', 'Student', 'Agriculture Specialist', 'Retired')
        AND t.transaction_date >= date('now', '-180 days')
  GROUP BY c.id HAVING avg_txn_amount > 2000000 OR large_txns > 0 ORDER BY avg_txn_amount DESC;
"""

    def _create_prompt(self, user_query: str) -> str:
        """Create a focused prompt for high-quality SQL generation with multilingual support."""

        # Detect query language for better context
        language_context = self._detect_query_language(user_query)

        return f"""You are a multilingual banking SQL specialist. Generate a precise SQLite query from the user's request.

{self.schema_context}

MULTILINGUAL QUERY EXAMPLES:
English: "Show all clients from Tashkent" → SELECT c.client_number, c.name, c.region FROM clients c WHERE c.region = 'Tashkent'
Russian: "Покажи всех клиентов из Ташкента" → SELECT c.client_number, c.name, c.region FROM clients c WHERE c.region = 'Tashkent'
Uzbek: "Toshkent mijozlarini ko'rsat" → SELECT c.client_number, c.name, c.region FROM clients c WHERE c.region = 'Tashkent'

CURRENCY EXAMPLES:
English: "accounts over 50,000 UZS" → WHERE ABS(a.balance) > 5000000
Russian: "счета свыше 50,000 сум" → WHERE ABS(a.balance) > 5000000
Uzbek: "50,000 so'm dan yuqori hisoblar" → WHERE ABS(a.balance) > 5000000

MULTILINGUAL BANKING TERMS MAPPING:
- баланс/balans = balance
- операция/operatsiya = transaction
- клиент/mijoz = client
- счет/hisob = account
- филиал/filial = branch
- суммa/summa/miqdor = amount
- риск/xavf = risk
- большой/katta = large
- общий/jami/umumiy = total
- показать/ko'rsatish = show/display

CORE INSTRUCTIONS:
1. Output ONLY the SQL query - no explanations, formatting, or extra text
2. Use proper SQLite syntax with banking field names
3. Always use table aliases: c=clients, a=accounts, t=transactions, b=branches
4. Include appropriate JOINs for multi-table queries
5. Add LIMIT 100 for queries that could return many rows
6. Use proper date functions: date('now'), date('now', '-30 days')
7. Return only SELECT statements
8. Handle multilingual terms (English/Russian/Uzbek) and map them to database fields

BANKING QUERY PATTERNS:
- Client queries: "SELECT c.client_number, c.name, c.region FROM clients c WHERE..."
- Account queries: "SELECT a.account_number, a.balance, c.name FROM accounts a JOIN clients c..."
- Transaction queries: "SELECT t.transaction_reference, t.amount, t.transaction_date FROM transactions t..."
- Regional queries: Use exact names: 'Tashkent', 'Samarkand', 'Bukhara', 'Andijan', 'Namangan', 'Fergana', 'Nukus'
- Amount filters: Use ABS(t.amount) > value for absolute amounts (amounts are stored in tiyin)
- Date filters: Use t.transaction_date >= date('now', '-30 days') format
- Status filters: Use a.status = 'ACTIVE' for active accounts

BANKING CONTEXT:
- Large amounts: > 5000000 tiyin (50,000 UZS / 50,000 so'm)
- High-risk: risk_score > 0.8 OR flagged_for_review = 1
- Cash transactions: transaction_subtype LIKE '%CASH%'
- Recent timeframe: last 30 days unless specified
- Currency precision: All amounts stored in tiyin (1 UZS = 100 tiyin)

DETECTED LANGUAGE: {language_context}

USER REQUEST: {user_query}

SQL Query:"""

    def _detect_query_language(self, user_query: str) -> str:
        """Detect the primary language of the user query."""
        query_lower = user_query.lower()

        # Cyrillic characters indicate Russian
        cyrillic_count = sum(1 for char in query_lower if '\u0400' <= char <= '\u04ff')

        # Common Russian banking terms
        russian_terms = [
            'клиент', 'счет', 'баланс', 'операция', 'филиал', 'банк', 'деньги',
            'покажи', 'найди', 'все', 'сумма', 'риск', 'большой', 'общий',
            'ташкент', 'самарканд', 'бухара', 'андижан', 'наманган', 'фергана'
        ]

        # Common Uzbek banking terms
        uzbek_terms = [
            'mijoz', 'hisob', 'balans', 'operatsiya', 'filial', 'bank', 'pul',
            'ko\'rsat', 'toping', 'barchasi', 'miqdor', 'xavf', 'katta', 'jami',
            'toshkent', 'samarqand', 'buxoro', 'andijon', 'namangan', 'farg\'ona'
        ]

        # Common English banking terms
        english_terms = [
            'client', 'account', 'balance', 'transaction', 'branch', 'bank', 'money',
            'show', 'find', 'all', 'amount', 'risk', 'large', 'total',
            'tashkent', 'samarkand', 'bukhara', 'andijan', 'namangan', 'fergana'
        ]

        # Count term matches
        russian_matches = sum(1 for term in russian_terms if term in query_lower)
        uzbek_matches = sum(1 for term in uzbek_terms if term in query_lower)
        english_matches = sum(1 for term in english_terms if term in query_lower)

        # Determine language based on matches and script
        if cyrillic_count > 0 or russian_matches > uzbek_matches and russian_matches > english_matches:
            return "Russian (Русский)"
        elif uzbek_matches > english_matches and uzbek_matches > russian_matches:
            return "Uzbek (O'zbek)"
        else:
            return "English"

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
        except Exception as e:
            return False, f"SQL syntax error: {str(e)}"

        # Banking-specific validations
        return self._validate_banking_rules(sql_query)

    def _validate_banking_rules(self, sql_query: str) -> Tuple[bool, str]:
        """Validate banking business rules and constraints."""
        upper_query = sql_query.upper()

        # Check for valid table names
        valid_tables = ['CLIENTS', 'ACCOUNTS', 'TRANSACTIONS', 'BRANCHES', 'PRODUCTS']
        tables_in_query = []
        for table in valid_tables:
            if table in upper_query:
                tables_in_query.append(table.lower())

        if not tables_in_query:
            return False, "Query must reference at least one valid banking table"

        # Amount validation - warn about unrealistic values
        if 'AMOUNT' in upper_query:
            # Check for realistic banking amounts (not too high to cause issues)
            import re
            amount_patterns = re.findall(r'AMOUNT\s*[><=]+\s*(\d+)', upper_query)
            for amount in amount_patterns:
                if int(amount) > settings.max_single_transaction_amount * 100:  # 100x max limit
                    logger.warning(f"Query contains very large amount filter: {amount}")

        # Date validation - ensure proper date functions
        if any(date_word in upper_query for date_word in ['DATE', 'DATETIME', 'NOW']):
            if "DATE('NOW'" not in upper_query and "DATETIME('NOW'" not in upper_query and 'TRANSACTION_DATE' in upper_query:
                # Check if using proper SQLite date functions
                if not any(func in upper_query for func in ["DATE(", "DATETIME(", "STRFTIME("]):
                    logger.warning("Query may need proper SQLite date functions")

        # Currency validation
        if 'CURRENCY' in upper_query:
            valid_currencies = ['UZS', 'USD', 'EUR']
            # Extract currency values from query
            import re
            currency_matches = re.findall(r"CURRENCY\s*=\s*['\"]([A-Z]{3})['\"]", upper_query)
            for currency in currency_matches:
                if currency not in valid_currencies:
                    return False, f"Invalid currency '{currency}'. Valid currencies: {valid_currencies}"

        # Region validation
        valid_regions = ['TASHKENT', 'SAMARKAND', 'BUKHARA', 'ANDIJAN', 'NAMANGAN', 'FERGANA', 'NUKUS']
        if 'REGION' in upper_query:
            import re
            region_matches = re.findall(r"REGION\s*=\s*['\"]([A-Za-z]+)['\"]", upper_query)
            for region in region_matches:
                if region.upper() not in valid_regions:
                    logger.warning(f"Region '{region}' may not exist in database. Valid regions: {[r.title() for r in valid_regions]}")

        return True, "Valid banking SQL query"

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
        """Generate SQL query from natural language input with fallback mechanisms."""
        logger.info(f"Generating SQL for query: {user_query}")

        # Try progressive query building for complex queries first
        fallback_result = self._try_fallback_mechanisms(user_query)
        if fallback_result:
            return fallback_result

        # Create prompt and call LLM
        prompt = self._create_prompt(user_query)
        raw_response = self._call_ollama(prompt)

        if not raw_response:
            return self._handle_failed_generation(user_query, "Failed to get response from LLM")

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
            # Try fallback mechanisms for failed queries
            fallback_result = self._handle_failed_query(user_query, sql_query, validation_message)
            if fallback_result:
                return fallback_result

            result["error"] = validation_message
            logger.warning(f"Generated invalid SQL: {validation_message}")
        else:
            logger.info(f"Generated valid SQL: {sql_query}")

        return result

    def _try_fallback_mechanisms(self, user_query: str) -> Optional[Dict]:
        """Try template matching and simple query patterns before LLM."""
        query_lower = user_query.lower()

        # Simple banking query templates
        banking_templates = {
            "show all clients": "SELECT client_number, name, region, status FROM clients LIMIT 100;",
            "list clients": "SELECT client_number, name, region, status FROM clients LIMIT 100;",
            "show clients from": self._handle_region_query(user_query),
            "find accounts": "SELECT account_number, account_type, balance, currency FROM accounts WHERE status = 'ACTIVE' LIMIT 100;",
            "list accounts": "SELECT account_number, account_type, balance, currency FROM accounts WHERE status = 'ACTIVE' LIMIT 100;",
            "show transactions": "SELECT transaction_reference, amount, transaction_date, transaction_type FROM transactions ORDER BY transaction_date DESC LIMIT 100;",
            "recent transactions": "SELECT transaction_reference, amount, transaction_date, transaction_type FROM transactions WHERE transaction_date >= date('now', '-30 days') ORDER BY transaction_date DESC LIMIT 100;"
        }

        for pattern, template in banking_templates.items():
            if pattern in query_lower:
                if callable(template):
                    sql_query = template
                else:
                    sql_query = template

                if sql_query:
                    is_valid, validation_message = self._validate_sql(sql_query)
                    if is_valid:
                        return {
                            "success": True,
                            "sql_query": sql_query,
                            "user_query": user_query,
                            "validation_message": "Generated from banking template",
                            "fallback_used": "template_matching"
                        }

        return None

    def _handle_region_query(self, user_query: str) -> str:
        """Handle region-specific queries."""
        regions = ['tashkent', 'samarkand', 'bukhara', 'andijan', 'namangan', 'fergana', 'nukus']
        query_lower = user_query.lower()

        for region in regions:
            if region in query_lower:
                return f"SELECT client_number, name, region, status FROM clients WHERE region = '{region.title()}' LIMIT 100;"

        return ""

    def _handle_failed_generation(self, user_query: str, error_message: str) -> Dict:
        """Handle cases where LLM fails to generate any response."""
        return {
            "success": False,
            "error": error_message,
            "sql_query": None,
            "user_query": user_query,
            "fallback_attempted": True
        }

    def _handle_failed_query(self, user_query: str, failed_sql: str, error_message: str) -> Optional[Dict]:
        """Handle cases where generated SQL is invalid."""
        # Try to fix common SQL issues
        fixed_sql = self._attempt_sql_fix(failed_sql, error_message)

        if fixed_sql:
            is_valid, validation_message = self._validate_sql(fixed_sql)
            if is_valid:
                return {
                    "success": True,
                    "sql_query": fixed_sql,
                    "user_query": user_query,
                    "validation_message": f"Auto-fixed: {validation_message}",
                    "fallback_used": "sql_fix",
                    "original_error": error_message
                }

        # If fixing fails, try simpler template approach
        return self._try_fallback_mechanisms(user_query)

    def _attempt_sql_fix(self, sql_query: str, error_message: str) -> Optional[str]:
        """Attempt to fix common SQL syntax issues."""
        fixed_sql = sql_query

        # Common fixes
        if "syntax error" in error_message.lower():
            # Add missing semicolon
            if not fixed_sql.strip().endswith(';'):
                fixed_sql += ';'

            # Fix common quote issues
            fixed_sql = fixed_sql.replace("'", "'").replace("'", "'")  # Fix smart quotes

            # Fix table name case issues
            table_fixes = {
                'Clients': 'clients',
                'Accounts': 'accounts',
                'Transactions': 'transactions',
                'Branches': 'branches',
                'Products': 'products'
            }
            for wrong, correct in table_fixes.items():
                fixed_sql = fixed_sql.replace(wrong, correct)

        return fixed_sql if fixed_sql != sql_query else None

    def build_progressive_query(self, user_query: str) -> Dict:
        """Build complex queries progressively by breaking them into simpler parts."""
        query_lower = user_query.lower()

        # Detect complex query patterns
        complex_patterns = {
            "compliance": ["suspicious", "flagged", "aml", "risk", "compliance"],
            "analytics": ["trend", "analysis", "compare", "growth", "performance"],
            "multi_table": ["client", "account", "transaction", "branch"],
            "aggregation": ["total", "sum", "average", "count", "group by", "max", "min"]
        }

        detected_complexity = []
        for pattern_type, keywords in complex_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_complexity.append(pattern_type)

        # If query is complex, break it down
        if len(detected_complexity) >= 2:
            return self._handle_complex_query(user_query, detected_complexity)

        # Otherwise, use standard generation
        return self.generate_sql(user_query)

    def _handle_complex_query(self, user_query: str, complexity_types: List[str]) -> Dict:
        """Handle complex queries by breaking them into components."""
        query_lower = user_query.lower()

        # Banking compliance queries
        if "compliance" in complexity_types:
            return self._build_compliance_query(user_query)

        # Multi-table analytical queries
        if "analytics" in complexity_types and "multi_table" in complexity_types:
            return self._build_analytical_query(user_query)

        # Aggregation queries
        if "aggregation" in complexity_types:
            return self._build_aggregation_query(user_query)

        # Default to standard generation
        return self.generate_sql(user_query)

    def _build_compliance_query(self, user_query: str) -> Dict:
        """Build compliance-focused banking queries."""
        query_lower = user_query.lower()

        # High-risk client analysis
        if any(word in query_lower for word in ["high risk", "suspicious client"]):
            sql = """
            SELECT c.client_number, c.name, c.risk_rating,
                   COUNT(t.id) as suspicious_txns,
                   SUM(ABS(t.amount)) as total_amount
            FROM clients c
            JOIN accounts a ON c.id = a.client_id
            JOIN transactions t ON a.id = t.account_id
            WHERE c.risk_rating = 'HIGH' OR t.flagged_for_review = 1
            GROUP BY c.id
            HAVING suspicious_txns > 5
            ORDER BY total_amount DESC
            LIMIT 50;
            """

        # AML threshold violations
        elif any(word in query_lower for word in ["aml", "large cash", "threshold"]):
            sql = """
            SELECT t.transaction_reference, t.amount, t.transaction_date,
                   c.name, c.risk_rating, t.risk_score
            FROM transactions t
            JOIN accounts a ON t.account_id = a.id
            JOIN clients c ON a.client_id = c.id
            WHERE (t.amount > 20000000 AND t.transaction_type = 'DEPOSIT')
               OR t.risk_score > 0.8
            ORDER BY t.amount DESC, t.risk_score DESC
            LIMIT 100;
            """

        # Flagged transactions review
        else:
            sql = """
            SELECT t.transaction_reference, t.amount, t.transaction_date,
                   t.risk_score, t.review_status, c.name
            FROM transactions t
            JOIN accounts a ON t.account_id = a.id
            JOIN clients c ON a.client_id = c.id
            WHERE t.flagged_for_review = 1
            ORDER BY t.risk_score DESC, t.transaction_date DESC
            LIMIT 100;
            """

        is_valid, validation_message = self._validate_sql(sql.strip())
        return {
            "success": is_valid,
            "sql_query": sql.strip() if is_valid else None,
            "user_query": user_query,
            "validation_message": validation_message,
            "query_type": "compliance_analysis",
            "progressive_build": True
        }

    def _build_analytical_query(self, user_query: str) -> Dict:
        """Build analytical queries for banking performance analysis."""
        query_lower = user_query.lower()

        # Branch performance analysis
        if any(word in query_lower for word in ["branch", "performance", "volume"]):
            sql = """
            SELECT b.region, b.branch_name,
                   COUNT(DISTINCT a.id) as total_accounts,
                   COUNT(DISTINCT c.id) as total_clients,
                   COUNT(t.id) as total_transactions,
                   SUM(ABS(t.amount)) as total_volume,
                   AVG(ABS(t.amount)) as avg_transaction_size
            FROM branches b
            LEFT JOIN accounts a ON b.id = a.branch_id
            LEFT JOIN clients c ON a.client_id = c.id
            LEFT JOIN transactions t ON a.id = t.account_id
            WHERE t.transaction_date >= date('now', '-3 months')
            GROUP BY b.id
            ORDER BY total_volume DESC
            LIMIT 20;
            """

        # Currency distribution analysis
        elif any(word in query_lower for word in ["currency", "distribution", "multi"]):
            sql = """
            SELECT a.currency,
                   COUNT(*) as account_count,
                   SUM(a.balance) as total_balance,
                   AVG(a.balance) as avg_balance,
                   COUNT(DISTINCT a.client_id) as unique_clients
            FROM accounts a
            WHERE a.status = 'ACTIVE'
            GROUP BY a.currency
            ORDER BY total_balance DESC;
            """

        # Transaction trend analysis
        else:
            sql = """
            SELECT strftime('%Y-%m', t.transaction_date) as month,
                   t.transaction_type,
                   COUNT(*) as transaction_count,
                   SUM(ABS(t.amount)) as total_volume,
                   AVG(ABS(t.amount)) as avg_amount
            FROM transactions t
            WHERE t.transaction_date >= date('now', '-12 months')
            GROUP BY month, t.transaction_type
            ORDER BY month DESC, total_volume DESC;
            """

        is_valid, validation_message = self._validate_sql(sql.strip())
        return {
            "success": is_valid,
            "sql_query": sql.strip() if is_valid else None,
            "user_query": user_query,
            "validation_message": validation_message,
            "query_type": "analytical_report",
            "progressive_build": True
        }

    def _build_aggregation_query(self, user_query: str) -> Dict:
        """Build aggregation-focused queries."""
        query_lower = user_query.lower()

        # Client portfolio summary
        if any(word in query_lower for word in ["client", "portfolio", "summary"]):
            sql = """
            SELECT c.region, c.income_level,
                   COUNT(*) as client_count,
                   COUNT(a.id) as total_accounts,
                   SUM(a.balance) as total_balance,
                   AVG(a.balance) as avg_balance_per_client
            FROM clients c
            LEFT JOIN accounts a ON c.id = a.client_id AND a.status = 'ACTIVE'
            GROUP BY c.region, c.income_level
            ORDER BY total_balance DESC;
            """

        # Account type distribution
        elif any(word in query_lower for word in ["account", "type", "distribution"]):
            sql = """
            SELECT a.account_type, a.currency,
                   COUNT(*) as account_count,
                   SUM(a.balance) as total_balance,
                   MIN(a.balance) as min_balance,
                   MAX(a.balance) as max_balance,
                   AVG(a.balance) as avg_balance
            FROM accounts a
            WHERE a.status = 'ACTIVE'
            GROUP BY a.account_type, a.currency
            ORDER BY total_balance DESC;
            """

        # Default aggregation
        else:
            sql = """
            SELECT COUNT(DISTINCT c.id) as total_clients,
                   COUNT(DISTINCT a.id) as total_accounts,
                   COUNT(t.id) as total_transactions,
                   SUM(ABS(t.amount)) as total_volume
            FROM clients c
            LEFT JOIN accounts a ON c.id = a.client_id
            LEFT JOIN transactions t ON a.id = t.account_id;
            """

        is_valid, validation_message = self._validate_sql(sql.strip())
        return {
            "success": is_valid,
            "sql_query": sql.strip() if is_valid else None,
            "user_query": user_query,
            "validation_message": validation_message,
            "query_type": "aggregation_summary",
            "progressive_build": True
        }

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
        """Return professional banking analysis scenarios with multilingual examples."""
        return [
            # Basic multilingual examples
            "Show all clients from Tashkent",
            "Покажи всех клиентов из Ташкента",
            "Toshkent mijozlarini ko'rsat",
            "Find high balance accounts over 50,000 UZS",
            "Найди счета с большим балансом свыше 50,000 сум",
            "50,000 so'm dan yuqori balansli hisoblarni toping",

            # Executive Dashboard & KPI Queries
            "Show total assets under management by region with client count and average account balance",
            "Покажи общие активы под управлением по регионам с количеством клиентов",
            "Mintaqa bo'yicha boshqaruvdagi jami aktivlarni mijozlar soni bilan ko'rsat",

            # Risk Management & Compliance
            "Identify high-risk transactions with multiple risk flags and client occupation analysis",
            "Найди операции с высоким риском и анализом по профессии клиентов",
            "Yuqori xavfli operatsiyalarni mijoz kasbi tahlili bilan aniqlang",

            # Client Behavior Analytics
            "Analyze banking channel preferences by client age group and tech-savvy occupations",
            "Проанализируй предпочтения банковских каналов по возрастным группам",
            "Yosh guruhlar bo'yicha bank kanallari afzalliklarini tahlil qiling",

            # Transaction Analysis
            "Show recent large transactions over 100,000 UZS",
            "Покажи недавние крупные операции свыше 100,000 сум",
            "100,000 so'm dan yuqori yaqinda bo'lgan katta operatsiyalarni ko'rsat",

            # Regional Analysis
            "Compare transaction volumes between Tashkent and Samarkand clients",
            "Сравни объемы операций между клиентами Ташкента и Самарканда",
            "Toshkent va Samarqand mijozlari o'rtasidagi operatsiya hajmlarini solishtir"
        ]


# Global LLM service instance
llm_service = LLMService()