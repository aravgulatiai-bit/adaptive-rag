@echo off
REM ===================================================================
REM  Adaptive RAG - one-click review demo
REM  Double-click this file. Press Enter to move through each step.
REM  Choose 1 in the menu for the full guided demo.
REM ===================================================================
cd /d "%~dp0"
title Adaptive RAG - Review Demo
mode con: cols=120 lines=45 >nul 2>&1
chcp 65001 >nul
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set TOKENIZERS_PARALLELISM=false
set PYTHONIOENCODING=utf-8

if not exist ".\.venv\Scripts\python.exe" (
  echo  Could not find .venv. Create it first:  python -m venv .venv  then  .venv\Scripts\pip install -e .
  pause
  exit /b 1
)
".\.venv\Scripts\python.exe" demo.py %*
echo.
pause
