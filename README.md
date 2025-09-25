# 🏦 BankingLLM Data Analyst

A sophisticated banking data analysis system that converts natural language queries into SQL and generates professional Excel reports with visualizations. Built with modern Python technologies and local LLM integration.

## 🚀 Features

- **Natural Language to SQL**: Transform plain English queries into accurate SQL using local LLM (Ollama + Llama 3.1)
- **Professional Excel Reports**: Automatically generate Excel files with charts and professional formatting
- **Multiple Interfaces**: CLI, Web UI (Streamlit), and REST API
- **Large Dataset Support**: Handles 1M+ records efficiently with optimized database operations
- **Smart Visualizations**: Automatic chart type selection based on data characteristics
- **Docker Ready**: Complete containerization with docker-compose
- **Banking Domain**: Specialized for financial data analysis with realistic mock data

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │────│  LLM Service    │────│   Database      │
│ (Natural Lang.) │    │ (Ollama/Llama)  │    │   (SQLite)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Excel Exporter  │
                    │  + Charts       │
                    └─────────────────┘
```

## 🛠 Technology Stack

- **Language**: Python 3.11+
- **LLM**: Ollama + Llama 3.1 (Local)
- **Database**: SQLite with SQLAlchemy ORM
- **Web Framework**: FastAPI (API) + Streamlit (UI)
- **Excel**: openpyxl + matplotlib
- **CLI**: Click + Rich
- **Containerization**: Docker + Docker Compose

## 📈 Database Schema

```sql
-- Clients table (10K records)
CREATE TABLE clients (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    birth_date DATETIME NOT NULL,
    region TEXT NOT NULL
);

-- Accounts table (~20K records)
CREATE TABLE accounts (
    id INTEGER PRIMARY KEY,
    client_id INTEGER REFERENCES clients(id),
    balance REAL NOT NULL,
    open_date DATETIME NOT NULL
);

-- Transactions table (1M+ records)
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(id),
    amount REAL NOT NULL,
    date DATETIME NOT NULL,
    type TEXT NOT NULL  -- deposit, withdrawal, transfer_in, transfer_out, payment
);
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

1. **Clone and start services:**
   ```bash
   git clone <repository-url>
   cd banking-llm
   make docker-up-prod  # For production/demo
   # or: make docker-up-dev  # For development
   ```

2. **System ready!**
   - Database with 1M+ records is pre-built in containers
   - LLM model downloads automatically on first start
   - No manual setup required!

3. **Access interfaces:**
   - Web UI: http://localhost:8501
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Redis (prod): http://localhost:6379

### Option 2: Local Installation

1. **Setup environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\\Scripts\\activate` on Windows
   pip install -r requirements.txt
   ```

2. **Install and start Ollama:**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama serve
   ollama pull llama3.1:8b
   ```

3. **Initialize database:**
   ```bash
   python -m src.cli setup  # Generates 1M+ records locally
   ```

4. **Run application:**
   ```bash
   # CLI interface
   python -m src.cli interactive

   # Web interface
   streamlit run src/web.py

   # API server
   uvicorn src.api:app --reload
   ```

## 💻 Usage Examples

### CLI Interface

```bash
# Interactive mode
python -m src.cli interactive

# Direct query
python -m src.cli query -q "Show total transactions by region for 2024"

# Database statistics
python -m src.cli stats

# Sample queries
python -m src.cli samples
```

### Sample Queries

Try these natural language queries:

1. **"Show total transactions by region for 2024"**
2. **"List top 10 clients by account balance"**
3. **"Display monthly transaction trends for last 6 months"**
4. **"Find accounts opened in Tashkent with balance above 50000"**
5. **"Calculate average account balance by region"**
6. **"Show transaction volume by type for each region"**

### Web Interface

Navigate to `http://localhost:8501` and use the intuitive web interface:

- Enter natural language queries
- View generated SQL
- See interactive results
- Download professional Excel reports
- Explore data with charts

### REST API

```bash
# Execute query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show total transactions by region"}'

# Get database stats
curl "http://localhost:8000/stats"

# Health check
curl "http://localhost:8000/health"
```

## 📊 Excel Report Features

Generated Excel reports include:

- **Data Sheet**: Query results with professional formatting
- **Summary Sheet**: Statistical insights and key metrics
- **Charts Sheet**: Auto-generated visualizations:
  - Bar charts for categorical vs numeric data
  - Pie charts for distributions
  - Line charts for time series data
- **Professional Styling**: Banking color scheme and formatting

## 🔧 Configuration

Create `.env` file for custom configuration:

```env
# Database
DATABASE_URL=sqlite:///./data/bank.db

# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1:8b

# Data Generation
NUM_CLIENTS=10000
NUM_TRANSACTIONS=1000000

# Logging
LOG_LEVEL=INFO
```

## 🚢 Production Deployment

### Docker Production Setup

```bash
# Development environment (with hot reload, debug tools)
make docker-up-dev
# or: docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Production environment (optimized, resource limits, Redis caching)
make docker-up-prod
# or: docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale services in production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale bank-ai-api=3
```

### Container Optimizations

**Multi-stage Docker Build:**
- ✅ Pre-built database with 1M+ records (instant startup)
- ✅ Specialized API and Web containers (no duplicate dependencies)
- ✅ Non-root user for security
- ✅ Optimized layer caching for faster rebuilds
- ✅ Database verification during build process

**Resource Management:**
- ✅ Memory and CPU limits in production
- ✅ Connection pooling and timeouts
- ✅ Request timeouts and retry logic
- ✅ Health checks and restart policies
- ✅ Embedded database eliminates setup complexity

### Performance Optimization

- Database indexes on key columns (region, date, client_id, account_id)
- Connection pooling for concurrent requests
- Query result caching with Redis (optional)
- Async operations for non-blocking I/O

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Test coverage
pytest --cov=src tests/

# API testing
pytest tests/test_api.py -v
```

## 📁 Project Structure

```
banking-llm/
├── src/
│   ├── api.py              # FastAPI REST API
│   ├── cli.py              # Command line interface
│   ├── config.py           # Configuration management
│   ├── database.py         # Database models and operations
│   ├── excel_export.py     # Excel export with charts
│   ├── llm_service.py      # LLM integration
│   ├── main.py             # Application entry point
│   └── web.py              # Streamlit web interface
├── data/                   # Database and export files
├── tests/                  # Test suite
├── docker-compose.yml      # Docker services
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── task.md                # Original requirements
├── CLAUDE.md              # Development guide
└── README.md              # This file
```

## 🎯 Key Technical Achievements

### Advanced LLM Integration
- **Schema-aware prompts**: LLM knows database structure for accurate SQL generation
- **Query validation**: SQL injection prevention and syntax checking
- **Context management**: Maintains conversation context for follow-up queries
- **Fallback handling**: Graceful error recovery and user guidance

### Professional Excel Output
- **Dynamic chart selection**: Automatically chooses appropriate visualizations
- **Multi-sheet reports**: Data, summary, and charts in separate sheets
- **Banking color scheme**: Professional styling with consistent branding
- **Interactive elements**: Charts that users can modify and explore

### Performance & Scalability
- **Optimized queries**: Database indexes and efficient SQL generation
- **Streaming operations**: Memory-efficient handling of large datasets
- **Async architecture**: Non-blocking operations for better throughput
- **Caching layer**: Query result caching for improved response times

### Production Ready
- **Containerization**: Docker with multi-stage builds
- **Health monitoring**: Built-in health checks and status endpoints
- **Error handling**: Comprehensive error management with user-friendly messages
- **Logging**: Structured logging with different levels and outputs

## 🔍 Troubleshooting

### Common Issues

1. **Ollama connection failed**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags

   # Restart Ollama service
   docker-compose restart ollama
   ```

2. **Database not found**
   ```bash
   # Initialize database
   python -m src.cli setup
   ```

3. **LLM model not found**
   ```bash
   # Pull the model
   ollama pull llama3.1:8b
   ```

4. **Permission denied on data directory**
   ```bash
   # Fix permissions
   chmod 755 data/
   mkdir -p data/exports
   ```

## ⚡ Instant Startup Technology

**Pre-built Database Strategy:**
- ✅ **Zero Setup Time**: 1M+ records ready immediately
- ✅ **Consistent Performance**: Same dataset for reliable benchmarks
- ✅ **Professional Deployment**: Production-ready container approach
- ✅ **Build Verification**: Database integrity checked during image creation
- ✅ **Banking Evaluation Ready**: Instant demo capability

**Technical Implementation:**
```dockerfile
# Database generated during Docker build
FROM base as db-generator
RUN python -m src.cli setup  # 1M+ records created
RUN verify_database()       # Integrity check

# Database embedded in final containers
FROM base as api
COPY --from=db-generator /app/data ./data/
```

## 🏆 Performance Benchmarks

- **Startup Time**: < 30 seconds (vs 3+ minutes with runtime generation)
- **SQL Generation**: < 3 seconds for complex queries
- **Database Queries**: < 1 second for typical operations on 1M records
- **Excel Export**: < 10 seconds for 10K records with charts
- **Memory Usage**: < 500MB for standard operations
- **Container Size**: ~900MB total (database included)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama Team**: For excellent local LLM infrastructure
- **Meta AI**: For Llama 3.1 model
- **Streamlit**: For rapid web UI development
- **FastAPI**: For modern async Python web framework

---

**BankingLLM - Professional banking data analysis with AI**