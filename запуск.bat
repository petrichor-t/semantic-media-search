@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  Semantic Media Search — Установка и запуск
echo ============================================
echo.

:: 1. Create virtual environment if missing
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Создание виртуального окружения .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo ОШИБКА: Не удалось создать виртуальное окружение.
        echo Убедитесь, что Python 3.10+ установлен и доступен в PATH.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Виртуальное окружение уже существует.
)

:: 2. Install/update dependencies
echo [2/3] Установка зависимостей ...
".venv\Scripts\python.exe" -m pip install -q --upgrade pip
".venv\Scripts\python.exe" -m pip install -q numpy sentence-transformers Pillow faiss-cpu PySide6
if errorlevel 1 (
    echo ОШИБКА: Не удалось установить зависимости.
    pause
    exit /b 1
)

:: 3. Create data directories
if not exist "data\database" mkdir "data\database"
if not exist "data\indexes" mkdir "data\indexes"

echo [3/3] Запуск приложения ...
echo.

set "PYTHONPATH=%~dp0src"
".venv\Scripts\python.exe" -B -m semantic_media_search.main
pause