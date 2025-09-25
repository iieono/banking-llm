#!/bin/bash
# Test script for pre-built database Docker build

set -e  # Exit on any error

echo "🏗️  Testing Bank AI LLM Docker Build with Pre-built Database"
echo "================================================================"

# Build images
echo "📦 Building Docker images with database generation..."
docker-compose build --parallel

echo ""
echo "✅ Build completed successfully!"

# Verify database exists in images
echo ""
echo "🔍 Verifying database exists in API container..."
docker run --rm bank-ai-llm-bank-ai-api python -c "
import sqlite3
import os
assert os.path.exists('data/bank.db'), 'Database file missing in API container'
conn = sqlite3.connect('data/bank.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM clients')
clients = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM accounts')
accounts = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM transactions')
transactions = cursor.fetchone()[0]
conn.close()
print(f'✅ API Container: {clients} clients, {accounts} accounts, {transactions} transactions')
assert clients >= 10000, f'Too few clients: {clients}'
assert accounts >= 15000, f'Too few accounts: {accounts}'
assert transactions >= 500000, f'Too few transactions: {transactions}'
"

echo ""
echo "🔍 Verifying database exists in Web container..."
docker run --rm bank-ai-llm-bank-ai-web python -c "
import sqlite3
import os
assert os.path.exists('data/bank.db'), 'Database file missing in Web container'
conn = sqlite3.connect('data/bank.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM transactions')
transactions = cursor.fetchone()[0]
conn.close()
print(f'✅ Web Container: {transactions} transactions ready')
assert transactions >= 500000, f'Too few transactions: {transactions}'
"

echo ""
echo "🎉 SUCCESS!"
echo "   ✅ Docker images built successfully"
echo "   ✅ Database pre-built and verified in both containers"
echo "   ✅ System ready for instant startup"
echo ""
echo "💡 Next steps:"
echo "   Run: make docker-up-prod"
echo "   Then: Open http://localhost:8501"
echo ""
echo "🏦 System is ready for banking evaluation!"