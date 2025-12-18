"""
Interactive ComfyUI UI on Modal (CPU-only).

Use this only when you need to edit/create workflows in the browser.
Close the tab when you're done to allow Modal to scale down to zero.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "comfyui-ui"

COMFY_PORT = 8188
COMFY_DIR = "/root/ComfyUI"
COMFY_OUTPUT_DIR = f"{COMFY_DIR}/output"

RESULTS_MOUNT = "/results"
PERSIST_MODELS_MOUNT = "/persist/models"
PERSIST_USER_MOUNT = "/persist/user"

app = modal.App(name=APP_NAME)

results_vol = modal.Volume.from_name("comfy-results", create_if_missing=True)
models_vol = modal.Volume.from_name("comfy-models", create_if_missing=True)
user_vol = modal.Volume.from_name("comfy-user", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PIP_ROOT_USER_ACTION": "ignore",
        }
    )
    .apt_install(
        "git",
        "curl",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install("httpx==0.28.1")
    .run_commands(
        f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git {COMFY_DIR}",
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio",
        f"python -m pip install -r {COMFY_DIR}/requirements.txt",
        f"git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git {COMFY_DIR}/custom_nodes/ComfyUI-Manager",
        f"python -m pip install -r {COMFY_DIR}/custom_nodes/ComfyUI-Manager/requirements.txt",
        f"git clone --depth 1 https://github.com/stavsap/comfyui-ollama.git {COMFY_DIR}/custom_nodes/comfyui-ollama",
        f"python -m pip install -r {COMFY_DIR}/custom_nodes/comfyui-ollama/requirements.txt",
    )
)


def _wait_for_comfy(timeout_s: int = 240) -> None:
    import httpx

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{COMFY_PORT}/api/system_stats", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("ComfyUI did not become ready in time")


def _ensure_symlink(target: Path, source: Path) -> None:
    source.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        return
    if target.exists():
        try:
            for item in target.iterdir():
                dest = source / item.name
                if dest.exists():
                    continue
                shutil.move(str(item), str(dest))
        except NotADirectoryError:
            pass
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=60 * 60,
    scaledown_window=60,
    max_containers=1,
    volumes={
        RESULTS_MOUNT: results_vol,
        PERSIST_MODELS_MOUNT: models_vol,
        PERSIST_USER_MOUNT: user_vol,
    },
)
@modal.concurrent(max_inputs=10)  # ComfyUI UI does concurrent API calls during load.
@modal.web_server(COMFY_PORT, startup_timeout=300)
def ui():
    _ensure_symlink(Path(f"{COMFY_DIR}/models"), Path(PERSIST_MODELS_MOUNT))
    _ensure_symlink(Path(f"{COMFY_DIR}/user"), Path(PERSIST_USER_MOUNT))

    Path(COMFY_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # For interactive use, allow Manager to use the network by default.
    Path(f"{COMFY_DIR}/user/__manager").mkdir(parents=True, exist_ok=True)
    (Path(f"{COMFY_DIR}/user/__manager/config.ini")).write_text(
        "[default]\nnetwork_mode = public\nfile_logging = False\n",
        encoding="utf-8",
    )

    subprocess.Popen(
        [
            "python",
            f"{COMFY_DIR}/main.py",
            "--cpu",
            "--listen",
            "0.0.0.0",
            "--port",
            str(COMFY_PORT),
        ],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    _wait_for_comfy()

