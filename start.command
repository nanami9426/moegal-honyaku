#!/bin/bash
# Finder 双击时工作目录不固定，始终从脚本所在目录启动。
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

UV_TMP_DIR=""
cleanup() {
    if [ -n "$UV_TMP_DIR" ]; then
        rm -rf "$UV_TMP_DIR"
    fi
}
on_error() {
    local exit_code=$?
    echo "[错误] 启动失败，请查看上方错误信息。"
    # 双击启动时保留终端，方便查看错误；非交互执行不会等待输入。
    if [ -t 0 ]; then
        read -r -p "按回车键退出..." || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT
trap on_error ERR

if [ "$(uname -s)" != "Darwin" ]; then
    echo "[错误] 此脚本用于 macOS；Windows 请运行 start.cmd。"
    false
fi
if [ "$(uname -m)" != "arm64" ]; then
    echo "[错误] 当前依赖要求 Apple Silicon Mac（M 系列芯片）。"
    echo "如果使用 M 系列 Mac，请关闭终端的 Rosetta 模式后重试。"
    false
fi

# 更新失败时直接使用已有解释器，跳过 uv、Python 下载和依赖同步。
if [ "${1:-}" = "--local" ] && [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    if [ ! -e .env ]; then
        cp .env.example .env
    fi
    echo "[信息] 使用已有本地环境启动：http://127.0.0.1:8000/docs"
    "$ROOT_DIR/.venv/bin/python" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
    exit $?
fi

# 将下载缓存和运行环境放在项目内，不修改系统 Python 或 shell 配置。
UV_BIN="$ROOT_DIR/.tools/uv/uv"
export UV_CACHE_DIR="$ROOT_DIR/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT_DIR/.python"
export UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv"
export UV_PYTHON_PREFERENCE=managed
export UV_PYTHON_INSTALL_BIN=0
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

if [ ! -x "$UV_BIN" ]; then
    echo "[信息] 下载项目本地 uv..."
    mkdir -p "$(dirname "$UV_BIN")"
    UV_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/moegal-uv.XXXXXX")"
    curl --fail --location --retry 3 \
        "https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz" \
        --output "$UV_TMP_DIR/uv.tar.gz"
    tar -xzf "$UV_TMP_DIR/uv.tar.gz" -C "$UV_TMP_DIR"
    install -m 755 "$UV_TMP_DIR/uv-aarch64-apple-darwin/uv" "$UV_BIN"
fi

echo "[信息] 准备 Python 3.12..."
"$UV_BIN" python install 3.12
echo "[信息] 安装项目依赖，首次运行可能需要较长时间..."
"$UV_BIN" sync --python 3.12 --default-index "$UV_DEFAULT_INDEX"

if [ ! -e .env ]; then
    cp .env.example .env
    echo "[信息] 已创建 .env，请填写翻译接口密钥后使用翻译功能。"
fi

echo "[信息] 启动服务：http://127.0.0.1:8000/docs"
echo "[信息] 首次启动会自动下载 OCR 模型；按 Ctrl+C 停止服务。"
"$UV_BIN" run --python 3.12 uvicorn app.main:app --host 0.0.0.0 --port 8000
