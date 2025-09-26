# 🏦 BankingLLM Data Analyst

Convert natural language to SQL and generate Excel reports for banking data. Ask questions in English, get SQL results and professional reports with charts.

## Features

- **Natural Language to SQL**: Ask banking questions in plain English
- **Professional Excel Reports**: Auto-generated reports with charts
- **1M+ Banking Records**: Realistic data ready to query (clients, accounts, transactions)
- **Instant Setup**: Docker with pre-built database, no configuration needed

## 🚀 Quick Setup

### Prerequisites

- Docker and Docker Compose
- Git

### Step 1: Clone and Setup
```bash
git clone <repository-url>
cd banking-llm

# Install dependencies (for local development)
pip install -r requirements.txt
```

### Step 2: Docker Setup

**The database is pre-built and ready to use - no setup required!**

**Production Setup:**
```bash
# Start production environment with optimized settings
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
docker exec banking-llm-ollama ollama pull qwen2.5:7b
```

**Quick Setup (Alternative):**
```bash
# Basic production setup
docker-compose up -d
docker exec banking-llm-ollama ollama pull qwen2.5:7b
```

**Access:** http://localhost:8501 (Web UI) • http://localhost:8000 (API)

### 🔒 Features
- **Instant Startup**: Pre-built database with 1M+ records ready to use
- **Multi-language Support**: qwen2.5:7b model with excellent Uzbek/Russian/English support
- **Professional Banking Data**: Realistic clients, accounts, and transaction patterns

### 🛡️ Database Protection
- **One-Time Generation**: Database is created once and never regenerated
- **Instant Docker Startup**: No 2-3 minute wait time for database generation
- **Persistent Storage**: Database persists across container restarts
- **Force Regeneration**: Use `--force-regenerate` flag only if needed (DANGEROUS)

### Verify Setup
Test that everything is working:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/stats
```

## How to Use

1. Go to http://localhost:8501
2. Type questions in the input box
3. View SQL results and download Excel reports

## 💼 Professional Banking Scenarios

The system now includes sophisticated banking scenarios with realistic client profiles, intelligent risk scoring, and professional business patterns:

### Executive Dashboard & KPIs
```
"Show total assets under management by region with client count and average account balance"
"Analyze transaction velocity by client occupation and identify top performing business segments"
"Calculate monthly revenue from fees by account type and show growth trends"
```

### Risk Management & AML
```
"Identify high-risk transactions with multiple risk flags and client occupation analysis"
"Show suspicious cash deposit patterns exceeding 5M UZS from business owners and restaurant operators"
"Find clients with unusual late-night transaction patterns and cross-reference with risk scores"
"Detect potential structuring patterns with multiple transactions just below reporting thresholds"
```

### Client Analytics & Segmentation
```
"Analyze banking channel preferences by client age group and tech-savvy occupations"
"Compare transaction volumes between IT executives vs traditional business owners"
"Identify VIP clients (ultra-high income) with multi-currency accounts and their preferred transaction types"
"Calculate customer lifetime value by analyzing transaction frequency and average amounts per occupation"
```

### Compliance & Regulatory
```
"Find clients with inconsistent transaction patterns relative to their declared occupation and income level"
"Identify international wire transfers over 10M UZS from import/export traders flagged for review"
"Show client acquisition trends by region with occupation distribution and income levels"
```

### 🎯 Key Data Features
- **16 Professional Occupations**: IT Executives, Bank Executives, Doctors, Business Owners, etc.
- **Sophisticated Risk Scoring**: Multi-factor algorithms based on real banking patterns
- **Realistic Financial Relationships**: Income-based account balances, occupation-specific transaction patterns
- **Advanced Business Intelligence**: Ready for impressive banking demonstrations

## Troubleshooting

### 🐳 Docker Troubleshooting

**Standard Docker Commands:**
```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f banking-llm-api
docker-compose logs -f ollama

# Stop containers
docker-compose down

# Restart containers
docker-compose restart
```

### Test API Directly
```bash
# Test a banking query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show total transactions by region"}'
```

---

Ready for banking data analysis, compliance reporting, and AI-powered financial queries.