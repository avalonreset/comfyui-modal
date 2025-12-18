"""
Unified ComfyUI on Modal (CPU-only).

This single Modal app provides:
- A browser UI (for creating/editing workflows).
- A headless runner API (for your SaaS backend).

Why both in one app? So you only manage one Modal app in the dashboard,
while still letting the headless runner scale to zero when idle.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import modal

APP_NAME = "comfyui"

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
    .pip_install("fastapi[standard]==0.115.4", "httpx==0.28.1")
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


def _sanitize_path_component(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Value cannot be empty")
    safe = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)[:128]


def _inject_ollama_url(workflow: dict[str, Any], ollama_url: str) -> int:
    injected = 0
    for node in workflow.values():
        class_type = str(node.get("class_type", ""))
        if "OllamaConnectivity" in class_type:
            inputs = node.setdefault("inputs", {})
            if isinstance(inputs, dict):
                inputs["url"] = ollama_url
                injected += 1
    return injected


def _submit_workflow(workflow: dict[str, Any]) -> str:
    import httpx

    payload = {"prompt": workflow, "client_id": uuid.uuid4().hex}
    r = httpx.post(f"http://127.0.0.1:{COMFY_PORT}/prompt", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["prompt_id"]


def _wait_for_history(prompt_id: str, timeout_s: int = 3600) -> dict[str, Any]:
    import httpx

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = httpx.get(f"http://127.0.0.1:{COMFY_PORT}/history/{prompt_id}", timeout=30)
        if r.status_code == 200:
            data = r.json().get(prompt_id)
            if isinstance(data, dict) and data.get("outputs"):
                return data
        time.sleep(2)
    raise TimeoutError("Workflow timed out")


def _extract_output_files(history_item: dict[str, Any]) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    outputs = history_item.get("outputs", {})
    if not isinstance(outputs, dict):
        return files

    for node_out in outputs.values():
        if not isinstance(node_out, dict):
            continue
        for items in node_out.values():
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                filename = it.get("filename")
                if not filename:
                    continue
                files.append(
                    {
                        "filename": str(filename),
                        "subfolder": str(it.get("subfolder", "")),
                        "type": str(it.get("type", "output")),
                    }
                )

    return files


def _copy_outputs_to_volume(
    *,
    user_id: str,
    job_id: str,
    output_files: list[dict[str, str]],
) -> list[str]:
    user_id_safe = _sanitize_path_component(user_id)
    job_id_safe = _sanitize_path_component(job_id)

    dest_dir = Path(RESULTS_MOUNT) / user_id_safe / job_id_safe
    dest_dir.mkdir(parents=True, exist_ok=True)

    stored: list[str] = []
    for f in output_files:
        if f.get("type") != "output":
            continue
        filename = Path(f["filename"]).name
        subfolder = f.get("subfolder", "")
        src = (Path(COMFY_OUTPUT_DIR) / subfolder / filename).resolve()
        if not src.exists():
            continue
        dest = dest_dir / filename
        shutil.copy2(src, dest)
        stored.append(f"{user_id_safe}/{job_id_safe}/{filename}")

    return stored


def _pick_primary_video(stored_paths: list[str]) -> str | None:
    video_exts = (".mp4", ".webm", ".mov", ".mkv", ".gif")
    for p in stored_paths:
        if p.lower().endswith(video_exts):
            return p
    return stored_paths[0] if stored_paths else None


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
@modal.concurrent(max_inputs=10)
@modal.web_server(COMFY_PORT, startup_timeout=300)
def ui():
    _ensure_symlink(Path(f"{COMFY_DIR}/models"), Path(PERSIST_MODELS_MOUNT))
    _ensure_symlink(Path(f"{COMFY_DIR}/user"), Path(PERSIST_USER_MOUNT))
    Path(COMFY_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

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


@app.cls(
    image=image,
    cpu=4,
    memory=8192,
    timeout=60 * 60,
    scaledown_window=60,
    volumes={
        RESULTS_MOUNT: results_vol,
        PERSIST_MODELS_MOUNT: models_vol,
        PERSIST_USER_MOUNT: user_vol,
    },
)
@modal.concurrent(max_inputs=1)
class ComfyRunner:
    @modal.enter()
    def enter(self) -> None:
        _ensure_symlink(Path(f"{COMFY_DIR}/models"), Path(PERSIST_MODELS_MOUNT))
        _ensure_symlink(Path(f"{COMFY_DIR}/user"), Path(PERSIST_USER_MOUNT))
        Path(COMFY_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        Path(f"{COMFY_DIR}/user/__manager").mkdir(parents=True, exist_ok=True)
        (Path(f"{COMFY_DIR}/user/__manager/config.ini")).write_text(
            "[default]\nnetwork_mode = offline\nfile_logging = False\n",
            encoding="utf-8",
        )

    def _ensure_comfy_running(self) -> None:
        import httpx

        try:
            r = httpx.get(f"http://127.0.0.1:{COMFY_PORT}/api/system_stats", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass

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

    @modal.fastapi_endpoint(method="POST")
    def run(self, body: dict[str, Any], request: "Request") -> dict[str, Any]:
        from fastapi import HTTPException, Request

        token = os.environ.get("COMFY_RUN_TOKEN", "").strip()
        if token:
            auth = ""
            if request is not None:
                auth = request.headers.get("authorization", "")
            if auth != f"Bearer {token}":
                raise HTTPException(status_code=401, detail="Unauthorized")

        user_id = str(body.get("user_id", "")).strip()
        ollama_url = str(body.get("ollama_url", "")).strip()
        workflow_json = body.get("workflow_json")

        if not user_id:
            raise ValueError("Missing required field: user_id")
        if not ollama_url:
            raise ValueError("Missing required field: ollama_url")
        if not isinstance(workflow_json, dict):
            raise ValueError("Missing required field: workflow_json (object)")

        job_id = uuid.uuid4().hex

        injected = _inject_ollama_url(workflow_json, ollama_url)
        if injected == 0:
            raise ValueError("No OllamaConnectivity node found in workflow_json")

        self._ensure_comfy_running()

        prompt_id = _submit_workflow(workflow_json)
        history_item = _wait_for_history(prompt_id, timeout_s=60 * 60)

        output_files = _extract_output_files(history_item)
        stored_paths = _copy_outputs_to_volume(
            user_id=user_id,
            job_id=job_id,
            output_files=output_files,
        )
        results_vol.commit()

        primary = _pick_primary_video(stored_paths)
        if not primary:
            raise RuntimeError("Workflow completed but no output files were found")

        return {
            "user_id": _sanitize_path_component(user_id),
            "job_id": job_id,
            "prompt_id": prompt_id,
            "result_path": primary,
            "stored_paths": stored_paths,
        }

