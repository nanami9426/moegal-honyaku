"""两端共用 Git 更新逻辑；只依赖标准库，更新失败不安装或修改运行环境。"""

import os
from pathlib import Path
import subprocess


# 只有实际更新成功才返回 0；其余情况让启动脚本复用本地环境。
USE_LOCAL = 10


def update_project(root: Path) -> int:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=15")

    def git(*args: str, check: bool = True, timeout=30):
        return subprocess.run(
            ["git", *args], cwd=root, env=env, check=check,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )

    try:
        # ZIP 解压目录即使位于另一个仓库内，也不能更新父级仓库。
        repo_root = Path(git("rev-parse", "--show-toplevel").stdout.strip())
        if repo_root.resolve() != root.resolve():
            print("[信息] 当前目录不是项目 Git 仓库，使用本地版本。", flush=True)
            return USE_LOCAL
        branch = git("symbolic-ref", "--quiet", "--short", "HEAD").stdout.strip()
        original = git("rev-parse", "HEAD").stdout.strip()
        if git("status", "--porcelain", "--untracked-files=all").stdout:
            print("[信息] 有本地改动或未跟踪文件，跳过更新，保留本地版本。", flush=True)
            return USE_LOCAL

        remote = git("config", "--get", f"branch.{branch}.remote").stdout.strip()
        remote_ref = git("config", "--get", f"branch.{branch}.merge").stdout.strip()
        print(f"[信息] 检查 {branch} 的上游更新（最多等待 60 秒）...", flush=True)
        # 下载完成之前不触碰工作区；不硬编码 main/master 或 origin。
        git("fetch", "--no-tags", "--", remote, remote_ref, timeout=60)
        target = git("rev-parse", "FETCH_HEAD").stdout.strip()
        if target == original:
            print("[信息] 已是最新版本，直接使用本地环境。", flush=True)
            return USE_LOCAL
        if git("merge-base", "--is-ancestor", original, target, check=False).returncode:
            print("[信息] 本地有独立提交或历史已分叉，跳过自动更新。", flush=True)
            return USE_LOCAL

        # 网络等待期间可能发生编辑或切换分支，合并前再次确认。
        if (
            git("rev-parse", "HEAD").stdout.strip() != original
            or git("symbolic-ref", "--quiet", "--short", "HEAD").stdout.strip() != branch
            or git("status", "--porcelain", "--untracked-files=all").stdout
        ):
            print("[信息] 检查更新期间本地状态发生变化，跳过更新。", flush=True)
            return USE_LOCAL
        # 禁止自动 stash 和覆盖忽略文件（例如 .env），不强制重置用户代码。
        # 本地合并不设置超时，避免中途终止 Git 导致工作区只更新一部分。
        git("merge", "--ff-only", "--no-autostash", "--no-overwrite-ignore", target, timeout=None)
        print(f"[信息] 已更新：{original[:8]} → {target[:8]}。", flush=True)
        return 0
    except FileNotFoundError:
        print("[提示] 未找到 Git，跳过更新，使用本地版本。", flush=True)
    except subprocess.TimeoutExpired:
        print("[提示] 更新超时，使用本地版本。", flush=True)
    except (subprocess.CalledProcessError, OSError):
        print("[提示] 无法更新，请检查 Git 仓库、上游分支、网络和访问权限；使用本地版本。", flush=True)
    return USE_LOCAL


if __name__ == "__main__":
    raise SystemExit(update_project(Path(__file__).resolve().parent.parent))
