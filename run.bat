@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
"%~dp0.venv\Scripts\python.exe" -m semantic_media_search.main
pause