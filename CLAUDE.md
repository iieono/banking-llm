# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BankingLLM Data Analyst - A professional banking data analysis system that converts natural language queries into SQL and generates Excel reports with visualizations. This project demonstrates advanced Python development skills with clean architecture and modern technologies.

## Architecture

### Core Components

- **`src/database.py`**: SQLAlchemy models (Clients, Accounts, Transactions) with efficient data generation for 1M+ records
- **`src/llm_service.py`**: Ollama/Llama integration with banking-specific prompt engineering and SQL validation
- **`src/excel_export.py`**: Professional Excel export engine with automatic chart generation
- **`src/api.py`**: FastAPI REST API with async operations and comprehensive error handling
- **`src/cli.py`**: Rich CLI interface with interactive mode and progress indicators
- **`src/web.py`**: Streamlit web interface with real-time query execution and data visualization
- **`src/config.py`**: Centralized configuration management with environment variable support

### Design Patterns

- **Clean Architecture**: Separation of concerns with clear boundaries between layers
- **Dependency Injection**: Configurable services for testing and flexibility
- **Factory Pattern**: Database and service instantiation
- **Strategy Pattern**: Different export formats and chart types

## Development Commands

### Local Development

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Initialize database (generates 1M+ records - takes ~2-3 minutes)
python -m src.cli setup

# Run different interfaces
python -m src.cli interactive          # Interactive CLI
streamlit run src/web.py               # Web UI on :8501
uvicorn src.api:app --reload          # API on :8000

# Database operations
python -m src.cli stats                # Show database statistics
python -m src.cli samples             # Show sample queries
```

### Docker Operations

```bash
# Production environment (optimized)
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Basic production setup
docker-compose up -d

# Pull LLM model (required first time)
docker exec banking-llm-ollama ollama pull qwen2.5:7b

# Verify database (optional - already pre-built)
curl http://localhost:8000/stats

# View logs
docker-compose logs -f bank-ai-api
docker-compose logs -f ollama

# Rebuild after code changes
docker-compose build --no-cache
docker-compose up -d
```

### Testing

```bash
# Run test suite (when implemented)
pytest tests/ -v

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/stats
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show total transactions by region"}'
```

## Key Technical Details

### Database Schema

The system uses a realistic banking schema with proper relationships:

- **clients**: 10K records with regions (Tashkent, Samarkand, Bukhara, etc.)
- **accounts**: ~20K records (1-3 accounts per client)
- **transactions**: 1M+ records with realistic amounts and types

**Pre-built Database**: Database is generated during Docker build with seed=42 for reproducible results. This ensures instant startup and consistent performance benchmarks.

Indexes on: region, client_id, account_id, date, balance for query performance.

### LLM Integration

- **Model**: Llama 3.1 8B via Ollama (runs locally)
- **Prompt Engineering**: Banking-specific prompts with database schema context
- **Safety**: SQL injection prevention, query validation, SELECT-only enforcement
- **Error Handling**: Graceful fallbacks and user-friendly error messages

### Excel Export Features

- **Dynamic Charts**: Bar, pie, and line charts based on data characteristics
- **Professional Styling**: Banking color scheme (#1f4e79 primary, #5b9bd5 secondary)
- **Multi-sheet**: Data, Summary, and Charts sheets
- **Smart Formatting**: Conditional formatting, alternating rows, auto-width columns

### Performance Optimizations

- **Database**: Batch inserts, optimized indexes, connection pooling
- **LLM**: Low temperature (0.1) for consistent SQL, response length limits
- **Memory**: Streaming operations for large datasets
- **Caching**: Query result caching (ready for Redis integration)

## Configuration

### Environment Variables

```env
# Database
DATABASE_URL=sqlite:///./data/bank.db

# LLM
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:7b

# Data Generation
NUM_CLIENTS=10000
NUM_TRANSACTIONS=1000000

# Logging
LOG_LEVEL=INFO

# Excel
EXCEL_OUTPUT_DIR=./data/exports
```

### Service Dependencies

- **Ollama**: Must be running on port 11434 with qwen2.5:7b model pulled
- **Database**: SQLite file created automatically in `data/` directory
- **Python**: 3.11+ required for modern type hints and async features

## Common Tasks

### Adding New Query Types

1. Update prompts in `llm_service.py` with new examples
2. Add validation rules for new SQL patterns
3. Test with sample queries in CLI/Web interface

### Adding New Chart Types

1. Extend `_add_charts()` method in `excel_export.py`
2. Add chart detection logic based on data characteristics
3. Update professional styling to match banking theme

### Database Schema Changes

1. Update SQLAlchemy models in `database.py`
2. Modify data generation logic
3. Update schema context in `llm_service.py`
4. Consider migration strategy for existing data

### Performance Tuning

1. Add database indexes for new query patterns
2. Implement query result caching in Redis
3. Optimize data generation with bulk operations
4. Monitor query execution times in logs

## Error Handling

### Common Issues

- **Ollama not running**: Check `http://localhost:11434/api/tags`
- **Model not found**: Run `ollama pull qwen2.5:7b`
- **Database empty**: Run setup command to generate data
- **Permission errors**: Check `data/` directory permissions

### Debugging

- **Logging**: Use loguru with structured logging throughout
- **SQL Validation**: Check generated queries in CLI with syntax highlighting
- **API Testing**: Use FastAPI auto-docs at `/docs` endpoint
- **Database Queries**: Direct SQL inspection via CLI stats command

## Code Quality Standards

### Python Style

- **Type Hints**: Full type annotations throughout
- **Async/Await**: Used in FastAPI and database operations
- **Error Handling**: Comprehensive try/catch with user-friendly messages
- **Documentation**: Docstrings for all classes and methods

### Architecture Principles

- **Single Responsibility**: Each module has clear, focused purpose
- **Dependency Inversion**: Services injected, not hardcoded
- **Open/Closed**: Easy to extend with new features
- **DRY**: Shared utilities and configuration

## Security Considerations

- **SQL Injection**: Parameterized queries, input validation
- **LLM Safety**: Query validation, SELECT-only enforcement
- **File Access**: Sandboxed export directory
- **API Security**: Input validation, rate limiting ready

## Deployment Notes

### Production Readiness

- **Docker**: Multi-stage builds for smaller images
- **Health Checks**: Built-in endpoints for monitoring
- **Logging**: Structured JSON logs for production
- **Configuration**: Environment-based configuration

### Scaling Considerations

- **Database**: Can migrate to PostgreSQL for production
- **LLM**: Multiple Ollama instances behind load balancer
- **Caching**: Redis for query result caching
- **API**: Horizontal scaling with load balancer

BankingLLM demonstrates professional Python development with modern patterns, clean architecture, and production-ready deployment while solving a real business problem in banking data analysis.