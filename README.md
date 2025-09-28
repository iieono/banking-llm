# Banking AI - Natural Language to SQL + Excel Reports

Ask banking questions → Get SQL + Excel charts

## 🚀 Quick Start

### Windows (Easy)
```batch
# Docker version
docker-run.bat

# Local version
run.bat
```

### Manual Setup

**Docker:**
```bash
docker-compose up -d
docker exec banking-llm-ollama ollama pull qwen2.5:14b
# Open: http://localhost:8505
```

**Local:**
```bash
pip install -r requirements.txt
ollama serve & ollama pull qwen2.5:14b
python -m src.cli setup --yes
python src/main.py web
# Open: http://localhost:8505
```

## 🧪 Test Options

**Web Interface:** `python src/main.py web` or `run.bat`
**CLI Interactive:** `python -m src.cli interactive`
**Direct Query:** `python -m src.cli query --query "your question"`

Try these simple queries:
- "Show all clients from Tashkent"
- "Count transactions by type"
- "Average account balance by region"
- "Top 5 clients by balance"
- "Show accounts opened this year"
- "List branches in each region"

## 📋 Example Queries & Results

See real examples of natural language → SQL → Excel workflow:

### Example 1: Simple Regional Analysis
- **Query**: "Average account balance by region"
- **Generated SQL**:
  ```sql
  SELECT c.region, AVG(a.balance/100.0) as average_balance_som
  FROM clients c
  JOIN accounts a ON c.id = a.client_id
  GROUP BY c.region;
  ```
- **Result**: `Clients_AverageByRegion__20250929_001248.xlsx`
- **What it shows**: 7 rows of clean regional data with clear bar charts perfect for quick analysis

### Example 2: Client Demographics
- **Query**: "Client distribution by income level and occupation"
- **Generated SQL**:
  ```sql
  SELECT income_level, occupation, COUNT(*) as client_count
  FROM clients
  GROUP BY income_level, occupation;
  ```
- **Result**: `Clients_CountByIncomeLevel_20250928_234857.xlsx`
- **What it shows**: 61 rows of client demographics with bar charts, pie charts, and distribution analysis

### Example 3: Complex Trend Analysis
- **Query**: "Account balance growth trends by client type"
- **Generated SQL**:
  ```sql
  SELECT c.name, c.occupation, a.balance/100.0 as balance_som,
         strftime('%Y-%m', t.transaction_date) as month
  FROM clients c
  JOIN accounts a ON c.id=a.client_id
  JOIN transactions t ON a.id=t.account_id
  WHERE c.status='ACTIVE' AND t.transaction_type='DEPOSIT'
  GROUP BY c.name, c.occupation, month
  ORDER BY month ASC;
  ```
- **Result**: `TransactionsByName_20250929_000506.xlsx`
- **What it shows**: 8,580 rows of time-series data with line charts, trend analysis, and multi-dimensional visualizations

### Features Demonstrated:
- ✅ Natural language processing
- ✅ Complex SQL generation with JOINs and time functions
- ✅ Professional Excel export with multiple chart types
- ✅ Time-series analysis and trend visualization
- ✅ Banking-specific data analysis
- ✅ Multilingual support (English/Russian/Uzbek)

## 📊 Charts

Excel downloads have 2 sheets:
- **Data**: Query results
- **Charts**: Auto-graphs (bar/pie/line)

Files save to `data/exports/` folder.

## 🔧 CLI Commands

```bash
python -m src.cli interactive    # Chat mode
python -m src.cli stats         # Database info
python -m src.cli samples       # Example queries
python -m src.cli setup         # Generate database
```

## ⚠️ Issues

**No AI response**: Check `ollama list` has qwen2.5:14b
**No data**: Run `python -m src.cli setup --yes`
**No charts**: Charts are in Excel file, not web interface

## Requirements

- Python 3.11+ OR Docker
- 8GB+ RAM
- 5GB disk space