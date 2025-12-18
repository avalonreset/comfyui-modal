# ComfyUI on Modal (Headless + Ollama)

This repo deploys a **CPU-only**, **headless** ComfyUI service on Modal for multi-user SaaS use cases.

## What you get

- A SaaS endpoint that accepts `{ user_id, ollama_url, workflow_json }`
  - Injects `ollama_url` into `OllamaConnectivity*` nodes (from `stavsap/comfyui-ollama`)
  - Runs the workflow via ComfyUI `/prompt` + `/history`
  - Copies outputs into a persistent Modal `Volume` mounted at `/results`
  - Returns `result_path` + `stored_paths`
- Persistent volumes for models and ComfyUI user state (mounted under `/persist/*` and symlinked into ComfyUI)
 - ComfyUI is started lazily (only when `run` is called) to minimize idle cost.

## Deploy

Prereqs:
- Python installed locally
- Modal CLI set up (`py -m pip install modal` then `py -m modal setup`)

Deploy:

```bash
py -m modal deploy modal_comfyui_headless.py
```

Modal will print two URLs:
- `...-run.modal.run` (your SaaS endpoint)

Tip: keep the endpoint private and set `COMFY_RUN_TOKEN` in Modal, then send `Authorization: Bearer <token>`.

## SaaS endpoint request

POST JSON:

```json
{
  "user_id": "user_123",
  "ollama_url": "https://<reachable-ollama-url>",
  "workflow_json": { "1": { "class_type": "...", "inputs": {} } }
}
```

Response includes:
- `result_path`: a path under the `/results` volume like `user_123/<job_id>/output.mp4`
- `stored_paths`: all output files copied to the volume
