from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)

    result = subprocess.run(
        args,
        cwd=cwd,
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        msg = (
            f"command failed: {' '.join(args)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        raise AssertionError(msg)
    return result


def _copy_worktree_files(repo: Path) -> None:
    scripts_dir = repo / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    shutil.copy2(REPO_ROOT / "Makefile", repo / "Makefile")
    shutil.copy2(REPO_ROOT / "scripts" / "worktree.sh", scripts_dir / "worktree.sh")
    shutil.copy2(REPO_ROOT / ".pre-commit-config.yaml", repo / ".pre-commit-config.yaml")
    shutil.copy2(REPO_ROOT / "requirements.in", repo / "requirements.in")
    shutil.copy2(REPO_ROOT / "requirements.txt", repo / "requirements.txt")
    shutil.copy2(REPO_ROOT / "dev-requirements.in", repo / "dev-requirements.in")
    shutil.copy2(REPO_ROOT / "dev-requirements.txt", repo / "dev-requirements.txt")


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "embedserve"
    repo.mkdir()
    _copy_worktree_files(repo)

    _run(["git", "init", "-b", "main"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    _run(["git", "config", "user.name", "Test User"], cwd=repo)

    (repo / "README.md").write_text("temp repo\n", encoding="utf-8")
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-m", "initial"], cwd=repo)
    return repo


def _make(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["make", *args], cwd=repo, check=check)


def _create_fake_python(tmp_path: Path) -> tuple[Path, Path]:
    log_path = tmp_path / "bootstrap.log"
    fake_python = tmp_path / "fakepython"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"LOG_FILE='{log_path}'",
                'if [ "$#" -eq 3 ] && [ "$1" = "-m" ] && [ "$2" = "venv" ]; then',
                '  target="$3"',
                '  mkdir -p "$target/bin"',
                "  cat > \"$target/bin/python\" <<'EOF'",
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"printf 'python %s\\n' \"$*\" >> '{log_path}'",
                'if [ "$#" -ge 2 ] && [ "$1" = "-m" ] && [ "$2" = "pip" ]; then',
                "  exit 0",
                "fi",
                'echo "unexpected python invocation: $*" >&2',
                "exit 1",
                "EOF",
                "  cat > \"$target/bin/pre-commit\" <<'EOF'",
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"printf 'pre-commit %s\\n' \"$*\" >> '{log_path}'",
                "exit 0",
                "EOF",
                '  chmod +x "$target/bin/python" "$target/bin/pre-commit"',
                "  exit 0",
                "fi",
                'echo "unexpected fake python invocation: $*" >&2',
                "exit 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IEXEC)
    return fake_python, log_path


def test_make_worktree_create_bootstraps_and_copies_env(tmp_path: Path) -> None:
    """Ensure `make worktree-create` bootstraps tooling, copies `.env`, and checks out the new branch at the same HEAD."""
    repo = _init_repo(tmp_path)
    fake_python, log_path = _create_fake_python(tmp_path)
    (repo / ".env").write_text("MODEL_ID=local\n", encoding="utf-8")

    _make(repo, "worktree-create", "WORKTREE=agent", f"PYTHON={fake_python}")

    worktree = tmp_path / "embedserve-agent"
    assert worktree.is_dir()
    assert (worktree / ".env").read_text(encoding="utf-8") == "MODEL_ID=local\n"
    assert (worktree / "venv" / "bin" / "python").is_file()
    assert _run(["git", "branch", "--show-current"], cwd=worktree).stdout.strip() == "agent"
    assert (
        _run(["git", "rev-parse", "HEAD"], cwd=worktree).stdout
        == _run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
        ).stdout
    )

    log_contents = log_path.read_text(encoding="utf-8")
    assert "python -m pip install --upgrade pip" in log_contents
    assert "python -m pip install -r requirements.txt -r dev-requirements.txt" in log_contents
    assert "pre-commit install" in log_contents


def test_make_worktree_create_fails_when_branch_exists(tmp_path: Path) -> None:
    """Reject worktree creation when the requested branch already exists, avoiding accidental branch reuse."""
    repo = _init_repo(tmp_path)
    _run(["git", "branch", "agent"], cwd=repo)

    result = _make(repo, "worktree-create", "WORKTREE=agent", "SETUP=0", check=False)

    assert result.returncode != 0
    assert "Branch already exists: agent" in result.stderr


def test_script_create_rolls_back_worktree_when_setup_fails(tmp_path: Path) -> None:
    """Roll back both the worktree and branch when the post-create setup command fails partway through."""
    repo = _init_repo(tmp_path)
    worktree = tmp_path / "embedserve-agent"

    result = _run(
        ["bash", "scripts/worktree.sh", "create"],
        cwd=repo,
        env={
            "WORKTREE_NAME": "agent",
            "SETUP": "1",
            "COPY_ENV": "0",
            "SETUP_COMMAND": 'printf "partial\\n" > "$WORKTREE_PATH/.bootstrapped"; false',
        },
        check=False,
    )

    assert result.returncode != 0
    assert not worktree.exists()
    assert _run(["git", "branch", "--list", "agent"], cwd=repo).stdout.strip() == ""
    worktree_list = _run(["git", "worktree", "list", "--porcelain"], cwd=repo).stdout
    assert str(worktree) not in worktree_list


def test_script_create_does_not_overwrite_existing_worktree_env(tmp_path: Path) -> None:
    """Copy the tracked `.env` into the worktree without leaking later uncommitted local-only edits."""
    repo = _init_repo(tmp_path)
    (repo / ".env").write_text("MODEL_ID=tracked\n", encoding="utf-8")
    _run(["git", "add", ".env"], cwd=repo)
    _run(["git", "commit", "-m", "track env"], cwd=repo)
    (repo / ".env").write_text("MODEL_ID=local-only\n", encoding="utf-8")

    _run(
        ["bash", "scripts/worktree.sh", "create"],
        cwd=repo,
        env={
            "WORKTREE_NAME": "agent",
            "SETUP": "1",
            "COPY_ENV": "1",
            "SETUP_COMMAND": 'printf "ready\n" > "$WORKTREE_PATH/.bootstrapped"',
        },
    )

    worktree = tmp_path / "embedserve-agent"
    assert (worktree / ".bootstrapped").read_text(encoding="utf-8") == "ready\n"
    assert (worktree / ".env").read_text(encoding="utf-8") == "MODEL_ID=tracked\n"


def test_make_worktree_remove_rejects_dirty_worktree_without_force(tmp_path: Path) -> None:
    """Refuse to remove a dirty worktree unless `FORCE=1` is provided, preserving uncommitted changes."""
    repo = _init_repo(tmp_path)
    _make(repo, "worktree-create", "WORKTREE=agent", "SETUP=0", "COPY_ENV=0")
    worktree = tmp_path / "embedserve-agent"
    (worktree / "README.md").write_text("dirty change\n", encoding="utf-8")

    result = _make(repo, "worktree-remove", "WORKTREE=agent", check=False)

    assert result.returncode != 0
    assert "Worktree has uncommitted changes" in result.stderr
    assert worktree.exists()


def test_make_worktree_remove_force_can_delete_branch(tmp_path: Path) -> None:
    """Allow forced removal to delete both a dirty worktree and its branch when explicitly requested."""
    repo = _init_repo(tmp_path)
    _make(repo, "worktree-create", "WORKTREE=agent", "SETUP=0", "COPY_ENV=0")
    worktree = tmp_path / "embedserve-agent"
    (worktree / "README.md").write_text("dirty change\n", encoding="utf-8")

    _make(
        repo,
        "worktree-remove",
        "WORKTREE=agent",
        "FORCE=1",
        "DELETE_BRANCH=1",
    )

    assert not worktree.exists()
    branch_result = _run(["git", "branch", "--list", "agent"], cwd=repo)
    assert branch_result.stdout.strip() == ""
    worktree_list = _run(["git", "worktree", "list", "--porcelain"], cwd=repo).stdout
    assert str(worktree) not in worktree_list
