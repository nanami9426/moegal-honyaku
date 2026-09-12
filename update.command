#!/bin/bash
# 整个流程先解析为函数，避免更新覆盖当前脚本时影响后续命令读取。
main() {
    local root_dir update_python
    root_dir="$(cd "$(dirname "$0")" && pwd)" || return 1
    cd "$root_dir" || return 1
    update_python="$root_dir/.venv/bin/python"
    if [ ! -x "$update_python" ]; then
        update_python="$(command -v python3)"
    fi
    if [ -n "$update_python" ] && "$update_python" "$root_dir/scripts/update_project.py"; then
        bash "$root_dir/start.command"
    else
        echo "[信息] 继续启动本地版本；已有环境无需联网安装依赖。"
        bash "$root_dir/start.command" --local
    fi
}
main "$@"
