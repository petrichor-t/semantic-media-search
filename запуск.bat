@echo off
cd /d "%~dp0"

echo ============================================
echo  Semantic Media Search - Setup and Run
echo ============================================
echo.

:: 1. Create virtual environment if missing
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [1/3] Creating Python virtual environment...
    python -m venv "%~dp0.venv"
    if errorlevel 1 (
        echo ERROR: Could not create .venv
        echo Check that Python 3.10+ is installed and in PATH.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment OK.
)

:: 2. Install dependencies
echo [2/3] Installing dependencies...
call "%~dp0.venv\Scripts\python.exe" -m pip install -q --upgrade pip
call "%~dp0.venv\Scripts\python.exe" -m pip install -q numpy "sentence-transformers" Pillow "faiss-cpu" PySide6
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

:: 3. Ensure data directories exist
if not exist "%~dp0data\database" mkdir "%~dp0data\database"
if not exist "%~dp0data\indexes" mkdir "%~dp0data\indexes"

echo [3/3] Starting application...
echo.

set "PYTHONPATH=%~dp0src"
call "%~dp0.venv\Scripts\python.exe" -B -m semantic_media_search.main
pause