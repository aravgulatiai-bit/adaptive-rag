@echo off
REM ===================================================================
REM  Adaptive RAG - one-click launcher for the web UI
REM  Double-click this file, wait ~60-90s for "Serving...", then open
REM  http://127.0.0.1:5000 in your browser.  Close this window to stop.
REM ===================================================================

cd /d "%~dp0"

REM Run models fully offline (skips the network reachability check).
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set TOKENIZERS_PARALLELISM=false

echo.
echo   Starting Adaptive RAG web UI...
echo   This can take ~60-90 seconds on first launch while models load.
echo   When you see "Serving Adaptive RAG UI", open:  http://127.0.0.1:5000
echo   (Close this window to stop the server.)
echo.

".\.venv\Scripts\python.exe" -m adaptive_rag.webapp --host 127.0.0.1 --port 5000

echo.
echo   Server stopped. Press any key to close.
pause >nul
