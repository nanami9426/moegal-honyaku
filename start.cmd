@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1

echo ==========================================
echo Moegal Honyaku startup script
echo ==========================================

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"
if errorlevel 1 (
    echo [ERROR] Cannot enter project directory.
    call :pause_on_error
    exit /b 1
)

set "TOOLS_DIR=%ROOT_DIR%.tools"
set "UV_HOME=%TOOLS_DIR%\uv"
set "UV_BIN=%UV_HOME%\uv.exe"
set "UV_TMP=%TEMP%\moegal_honyaku_uv.zip"
set "UV_TMP_DIR=%TEMP%\moegal_honyaku_uv"

set "UV_CACHE_DIR=%ROOT_DIR%.cache\uv"
set "UV_PYTHON_INSTALL_DIR=%ROOT_DIR%.python"
set "UV_PROJECT_ENVIRONMENT=%ROOT_DIR%.venv"
set "UV_PYTHON_PREFERENCE=managed"
set "UV_PYTHON_INSTALL_BIN=0"

if not defined UV_DEFAULT_INDEX set "UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple"
echo [INFO] uv index: %UV_DEFAULT_INDEX%

call :ensure_uv_local
if errorlevel 1 exit /b 1

echo [INFO] Installing project-local Python 3.12 ...
set "UV_PYTHON_INSTALL_ARGS=3.12"
call "%UV_BIN%" python install --help | findstr /i /c:"--no-bin" >nul
if not errorlevel 1 set "UV_PYTHON_INSTALL_ARGS=3.12 --no-bin"
call "%UV_BIN%" python install %UV_PYTHON_INSTALL_ARGS%
if errorlevel 1 (
    echo [ERROR] Failed to install local Python.
    call :pause_on_error
    exit /b 1
)

echo [INFO] Running uv sync ...
call "%UV_BIN%" sync --python 3.12 --default-index "%UV_DEFAULT_INDEX%"
if errorlevel 1 (
    echo [ERROR] uv sync failed.
    call :pause_on_error
    exit /b 1
)

echo [INFO] Starting service ...
call "%UV_BIN%" run --python 3.12 uvicorn main:app --host 0.0.0.0 --port 8000
set "EXIT_CODE=%ERRORLEVEL%"
exit /b %EXIT_CODE%


:ensure_uv_local
if exist "%UV_BIN%" (
    echo [INFO] Found local uv: %UV_BIN%
    goto :eof
)

echo [INFO] Local uv not found, downloading ...
set "PS_BIN="
where powershell >nul 2>&1
if not errorlevel 1 set "PS_BIN=powershell"
if not defined PS_BIN (
    where pwsh >nul 2>&1
    if not errorlevel 1 set "PS_BIN=pwsh"
)
if not defined PS_BIN (
    echo [ERROR] PowerShell not found: powershell or pwsh. Cannot download uv.
    call :pause_on_error
    exit /b 1
)

set "UV_ARCH=x86_64"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "UV_ARCH=aarch64"
if /i "%PROCESSOR_ARCHITEW6432%"=="ARM64" set "UV_ARCH=aarch64"
set "UV_ZIP_URL=https://github.com/astral-sh/uv/releases/latest/download/uv-%UV_ARCH%-pc-windows-msvc.zip"

if exist "%UV_HOME%" rd /s /q "%UV_HOME%"
if not exist "%UV_HOME%" mkdir "%UV_HOME%"

if exist "%UV_TMP%" del /f /q "%UV_TMP%" >nul 2>&1
if exist "%UV_TMP_DIR%" rd /s /q "%UV_TMP_DIR%"
mkdir "%UV_TMP_DIR%" >nul 2>&1

echo [INFO] Download uv: %UV_ZIP_URL%
%PS_BIN% -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -UseBasicParsing -Uri '%UV_ZIP_URL%' -OutFile '%UV_TMP%'"
if errorlevel 1 (
    echo [ERROR] uv download failed.
    call :pause_on_error
    exit /b 1
)

%PS_BIN% -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%UV_TMP%' -DestinationPath '%UV_TMP_DIR%' -Force"
if errorlevel 1 (
    echo [ERROR] uv extract failed.
    call :pause_on_error
    exit /b 1
)

set "UV_EXTRACTED="
for /r "%UV_TMP_DIR%" %%F in (uv.exe) do (
    if not defined UV_EXTRACTED set "UV_EXTRACTED=%%F"
)

if not defined UV_EXTRACTED (
    echo [ERROR] uv.exe not found in downloaded archive.
    call :pause_on_error
    exit /b 1
)

copy /y "%UV_EXTRACTED%" "%UV_BIN%" >nul
if errorlevel 1 (
    echo [ERROR] Cannot write local uv.exe.
    call :pause_on_error
    exit /b 1
)

del /f /q "%UV_TMP%" >nul 2>&1
rd /s /q "%UV_TMP_DIR%" >nul 2>&1
echo [INFO] Local uv is ready: %UV_BIN%
goto :eof

:pause_on_error
echo.
pause
goto :eof
