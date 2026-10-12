@echo off
title Enterprise Document Intelligence & Advanced RAG System
cd /d "%~dp0"
echo =======================================================
echo  Enterprise Document Intelligence & Advanced RAG System
echo  Opening Dashboard at http://127.0.0.1:5000 ...
echo =======================================================
timeout /t 2 /nobreak >nul
start http://127.0.0.1:5000
python run_server.py
pause
