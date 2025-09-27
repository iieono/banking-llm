@echo off
REM Unified startup script for BankingLLM system
set PYTHONIOENCODING=utf-8
call venv\Scripts\activate
echo.
echo ========================================
echo  Banking Database Analysis Tool - Starting...
echo ========================================
echo.

REM Check if database exists
if not exist "data\bank.db" (
    echo Database not found. Creating database with sample data...
    echo This may take 2-3 minutes...
    python -m src.cli setup
    echo.
)

REM Check if Ollama is running
echo Checking Ollama service...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Ollama is not running!
    echo For LLM-powered queries, please:
    echo   1. Install Ollama from https://ollama.com/download
    echo   2. Run: ollama pull qwen2.5:14b
    echo   3. Ollama will start automatically
    echo.
    echo The analysis tool will work without Ollama, but smart SQL generation will be unavailable.
    echo.
    pause
)

echo.
echo Starting Banking Database Analysis Tool...
echo.
echo  Analysis Interface: http://localhost:8505
echo  Features: Read-only analysis, Professional Excel reports, Banking intelligence
echo  Database: %CD%\data\bank.db
echo.
echo Press Ctrl+C to stop the application.
echo ========================================
echo.

REM Start main application (defaults to web interface)
python src/main.py web