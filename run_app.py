#!/usr/bin/env python3
"""
Enterprise Control Plane launcher.
Starts the FastAPI backend and Next.js frontend together for local development.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import os
from pathlib import Path


ROOT_DIR = Path(__file__).parent
BACKEND_PORT = "8000"
FRONTEND_PORT = "3000"


def check_python_version() -> bool:
    if sys.version_info < (3, 10):
        print("Python 3.10 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    return True


def ensure_frontend_dependencies() -> bool:
    package_json = ROOT_DIR / "package.json"
    node_modules = ROOT_DIR / "node_modules"

    if not package_json.exists():
        print("Missing package.json for the Next.js frontend.")
        return False

    if node_modules.exists():
        return True

    npm_cmd = shutil.which("npm")
    if not npm_cmd:
        print("npm was not found on PATH.")
        return False

    print("Installing frontend dependencies...")
    try:
        subprocess.check_call([npm_cmd, "install"], cwd=ROOT_DIR)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Frontend dependency installation failed: {exc}")
        return False


def create_directories() -> None:
    for directory in ["data", "models", "logs", "output", "chroma_db", "frontend"]:
        (ROOT_DIR / directory).mkdir(parents=True, exist_ok=True)


def start_processes() -> int:
    npm_cmd = shutil.which("npm")
    if not npm_cmd:
        print("npm was not found on PATH.")
        return 1

    frontend_env = os.environ.copy()
    frontend_env.setdefault("NODE_OPTIONS", "--max-old-space-size=4096")

    backend_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.web_app:app",
            "--host",
            "0.0.0.0",
            "--port",
            BACKEND_PORT,
            "--reload",
        ],
        cwd=ROOT_DIR,
    )

    frontend_process = subprocess.Popen(
        [npm_cmd, "run", "dev", "--", "--turbo", "--port", FRONTEND_PORT],
        cwd=ROOT_DIR,
        env=frontend_env,
    )

    print("=" * 72)
    print("Enterprise Control Plane")
    print("=" * 72)
    print(f"FastAPI backend : http://127.0.0.1:{BACKEND_PORT}")
    print(f"Next.js frontend: http://127.0.0.1:{FRONTEND_PORT}")
    print("Press Ctrl+C to stop both services.")

    try:
        while True:
            time.sleep(1)

            if backend_process.poll() is not None:
                print("Backend process exited unexpectedly.")
                frontend_process.terminate()
                return backend_process.returncode or 1

            if frontend_process.poll() is not None:
                print("Frontend process exited unexpectedly.")
                backend_process.terminate()
                return frontend_process.returncode or 1
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        for process in (frontend_process, backend_process):
            if process.poll() is None:
                process.terminate()

        for process in (frontend_process, backend_process):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    return 0


def main() -> None:
    if not check_python_version():
        sys.exit(1)

    create_directories()

    if not ensure_frontend_dependencies():
        sys.exit(1)

    sys.exit(start_processes())


if __name__ == "__main__":
    main()
