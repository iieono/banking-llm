@echo off
REM Docker startup script for BankingLLM system
echo.
echo ========================================
echo  BankingLLM Data Analyst - Docker Mode
echo ========================================
echo.
echo Starting services...
echo   - Ollama LLM Service
echo   - BankingLLM Web Interface
echo.

REM Start Docker services
docker-compose up -d

REM Wait a moment for services to start
echo Waiting for services to initialize...
timeout /t 10 /nobreak >nul

REM Show status
echo.
echo Service Status:
docker-compose ps

echo.
echo ========================================
echo  System Ready!
echo ========================================
echo.
echo  Web Interface: http://localhost:8501
echo  Features: Chat interface, Dark theme, Excel export
echo  Ollama LLM:     http://localhost:11434
echo.
echo To stop the system: docker-compose down
echo To view logs:       docker-compose logs -f
echo.
pause