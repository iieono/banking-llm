# Makefile for Bank AI LLM project

.PHONY: help install setup run-cli run-web run-api docker-build docker-up docker-down clean test

# Default target
help:
	@echo "🏦 BankingLLM Data Analyst"
	@echo ""
	@echo "📋 Local Development:"
	@echo "  install      Install Python dependencies"
	@echo "  setup        Initialize database with mock data (local only)"
	@echo "  run-cli      Run CLI in interactive mode"
	@echo "  run-web      Run Streamlit web interface"
	@echo "  run-api      Run FastAPI server"
	@echo ""
	@echo "🐳 Docker Commands:"
	@echo "  docker-up-dev    Start development environment (database pre-built)"
	@echo "  docker-up-prod   Start production environment (optimized)"
	@echo "  docker-build     Build optimized Docker images"
	@echo "  docker-verify    Verify pre-built database"
	@echo "  docker-down      Stop all Docker services"
	@echo ""
	@echo "🧹 Utilities:"
	@echo "  clean       Clean generated files"
	@echo "  check-clean Check if project is clean for git/Docker"
	@echo "  test        Run tests"
	@echo ""
	@echo "💡 Quick Start:"
	@echo "  make docker-up-prod  # For bank demo/evaluation"
	@echo "  make docker-up-dev   # For development"

# Install dependencies
install:
	pip install -r requirements.txt

# Setup database
setup:
	python -m src.cli setup

# Run CLI
run-cli:
	python -m src.cli interactive

# Run web interface
run-web:
	streamlit run src/web.py

# Run API server
run-api:
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Docker operations
docker-build:
	docker-compose build --parallel

docker-build-prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml build --parallel

# Development environment
docker-up-dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
	@echo "Waiting for services to start..."
	@sleep 10
	@echo "Pulling LLM model..."
	docker exec banking-llm-ollama ollama pull llama3.1:8b
	@echo ""
	@echo "🚀 Development environment ready!"
	@echo "   Database with 1M+ records is pre-built and ready"
	@echo ""
	@echo "Available services:"
	@echo "  - Web UI: http://localhost:8501"
	@echo "  - API: http://localhost:8000"
	@echo "  - API Docs: http://localhost:8000/docs"
	@echo "  - DB Admin: http://localhost:8080"
	@echo ""
	@echo "No database setup required - system ready for immediate use!"

# Production environment
docker-up-prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "Waiting for services to start..."
	@sleep 15
	@echo "Pulling LLM model..."
	docker exec banking-llm-ollama ollama pull llama3.1:8b
	@echo ""
	@echo "🏦 Production environment ready!"
	@echo "   Database with 1M+ records is pre-built and optimized"
	@echo ""
	@echo "Available services:"
	@echo "  - Web UI: http://localhost:8501"
	@echo "  - API: http://localhost:8000"
	@echo "  - API Docs: http://localhost:8000/docs"
	@echo "  - Redis Cache: http://localhost:6379"
	@echo ""
	@echo "✅ System ready for banking evaluation - no setup required!"

# Standard development (backward compatibility)
docker-up: docker-up-dev

docker-down:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.prod.yml down -v

docker-down-dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml down -v

docker-down-prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml down -v

# Verify database (database is pre-built, this just checks it)
docker-verify:
	@echo "Verifying pre-built database..."
	curl -s http://localhost:8000/stats | python -m json.tool

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf data/*.db data/*.sqlite data/*.sqlite3
	rm -rf data/exports/*.xlsx data/exports/*.csv
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -f *.log *.tmp *.temp

# Check if project is clean for git and Docker
check-clean:
	python check-clean.py

# Run tests
test:
	pytest tests/ -v

# Test Docker build with pre-built database
test-build:
	@echo "🧪 Testing Docker build process..."
	bash test-build.sh

# Test running system end-to-end
test-system:
	@echo "🧪 Testing running system..."
	python test-system.py

# Full system test (build + run + test)
test-full: docker-up-prod
	@echo "⏳ Waiting for system to start..."
	@sleep 20
	@$(MAKE) test-system

# Development setup (full setup for new developers)
dev-setup: install setup
	@echo "Development environment ready!"
	@echo "Run 'make run-cli', 'make run-web', or 'make run-api'"

# Docker development setup
docker-dev: docker-build docker-up docker-setup
	@echo "Docker development environment ready!"