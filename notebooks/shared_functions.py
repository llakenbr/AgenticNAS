# shared functions for notebooks
import os
import sys
from pathlib import Path

def find_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents]
    for candidate in candidates:
        if (candidate / "apis").exists() and (candidate / "n8n").exists():
            return candidate
    raise RuntimeError("Could not locate repo root from current working directory.")

def read_env_file(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return

def read_secret_file(path: Path) -> str:
    secret = ""
    if not path.exists():
        return
    secret = path.read_text(encoding="utf-8").strip()
    return secret