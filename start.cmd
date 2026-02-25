@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

echo ================================
echo Moegal Honyaku 服务启动脚本
echo ================================

call :detect_python
if not defined PYTHON_BIN (
    echo [INFO] 未检测到 Python 环境。
    choice /C YN /N /M "是否安装 Python 3.12？ [Y/N]: "
    if errorlevel 2 (
        echo [INFO] 用户取消安装，脚本退出。
        exit /b 0
    )

    where winget >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] 未找到 winget，无法自动安装 Python 3.12。
        echo 请先手动安装 Python 3.12 后重新运行脚本。
        exit /b 1
    )

    echo [INFO] 开始安装 Python 3.12 ...
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [ERROR] Python 3.12 安装失败。
        exit /b 1
    )

    call :detect_python_after_install
    if not defined PYTHON_BIN (
        echo [ERROR] Python 安装完成，但当前 CMD 会话仍未识别到 Python。
        echo 请关闭并重新打开 CMD 后再运行本脚本。
        exit /b 1
    )
)

call :get_python_version
if not defined PY_VER (
    echo [ERROR] 无法获取 Python 版本，脚本退出。
    exit /b 1
)

echo [INFO] 检测到 Python 版本：!PY_VER!

if !PY_MAJOR! LSS 3 call :warn_low_version
if !PY_MAJOR! EQU 3 if !PY_MINOR! LSS 9 call :warn_low_version

echo [INFO] 安装 uv ...
call "%PYTHON_BIN%" %PYTHON_ARGS% -m ensurepip --upgrade >nul 2>&1
call "%PYTHON_BIN%" %PYTHON_ARGS% -m pip install --user -U uv
if errorlevel 1 (
    echo [ERROR] uv 安装失败。
    exit /b 1
)

for /f "delims=" %%i in ('call "%PYTHON_BIN%" %PYTHON_ARGS% -m site --user-base 2^>nul') do set "USER_BASE=%%i"
if defined USER_BASE (
    if exist "!USER_BASE!\Scripts\uv.exe" set "PATH=!USER_BASE!\Scripts;!PATH!"
)

set "UV_BIN="
where uv >nul 2>&1 && set "UV_BIN=uv"
if not defined UV_BIN if defined USER_BASE if exist "!USER_BASE!\Scripts\uv.exe" set "UV_BIN=!USER_BASE!\Scripts\uv.exe"

if not defined UV_BIN (
    echo [ERROR] uv 已安装，但当前会话无法找到 uv 命令。
    echo 请关闭并重新打开 CMD 后重试。
    exit /b 1
)

echo [INFO] 执行 uv sync ...
call "!UV_BIN!" sync
if errorlevel 1 (
    echo [ERROR] uv sync 执行失败。
    exit /b 1
)

echo [INFO] 启动服务：uvicorn main:app --host 0.0.0.0 --port 5000
call "!UV_BIN!" run uvicorn main:app --host 0.0.0.0 --port 5000
exit /b %errorlevel%


:detect_python
set "PYTHON_BIN="
set "PYTHON_ARGS="
where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_BIN=python"
    goto :eof
)
where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_BIN=py"
    set "PYTHON_ARGS=-3"
    goto :eof
)
goto :eof

:detect_python_after_install
call :detect_python
if defined PYTHON_BIN goto :eof

if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    set "PYTHON_BIN=%LocalAppData%\Programs\Python\Python312\python.exe"
    set "PYTHON_ARGS="
    goto :eof
)
if exist "%ProgramFiles%\Python312\python.exe" (
    set "PYTHON_BIN=%ProgramFiles%\Python312\python.exe"
    set "PYTHON_ARGS="
    goto :eof
)
if exist "%ProgramFiles(x86)%\Python312\python.exe" (
    set "PYTHON_BIN=%ProgramFiles(x86)%\Python312\python.exe"
    set "PYTHON_ARGS="
    goto :eof
)
goto :eof

:get_python_version
set "PY_VER="
set "PY_MAJOR="
set "PY_MINOR="
set "PY_PATCH="

for /f "tokens=2 delims= " %%v in ('call "%PYTHON_BIN%" %PYTHON_ARGS% -V 2^>^&1') do set "PY_VER=%%v"
if not defined PY_VER goto :eof

for /f "tokens=1,2,3 delims=." %%a in ("%PY_VER%") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
    set "PY_PATCH=%%c"
)
goto :eof

:warn_low_version
echo [WARN] 当前 Python 版本低于 3.9（!PY_VER!）。
goto :eof
