@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"
if errorlevel 1 exit /b 1
set "UPDATE_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%UPDATE_PYTHON%" set "UPDATE_PYTHON=python"

rem 预先解析整个命令块，避免更新覆盖当前批处理文件后读到错误位置。
(
    "%UPDATE_PYTHON%" "%~dp0scripts\update_project.py"
    if errorlevel 1 (
        echo [INFO] Starting local version without reinstalling existing dependencies.
        call "%~dp0start.cmd" --local
    ) else (
        call "%~dp0start.cmd"
    )
    exit /b
)
