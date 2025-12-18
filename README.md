# ComfyUI on Modal (UI + Headless + Ollama)

This repo deploys **CPU-only** ComfyUI on Modal for multi-user SaaS use cases.

You can run:
- **UI mode** (browser) to build/test workflows
- **Headless mode** (API) to execute workflows for users

## What you get

- A headless SaaS endpoint that accepts `{ user_id, ollama_url, workflow_json }`
  - Injects `ollama_url` into `OllamaConnectivity*` nodes (from `stavsap/comfyui-ollama`)
  - Runs the workflow via ComfyUI `/prompt` + `/history`
  - Copies outputs into a persistent Modal `Volume` mounted at `/results`
  - Returns `result_path` + `stored_paths`
- Persistent volumes for models and ComfyUI user state (mounted under `/persist/*` and symlinked into ComfyUI)
 - ComfyUI is started lazily (only when `run` is called) to minimize idle cost.
 - Optional browser UI for workflow creation/testing

## Deploy

Prereqs:
- Python installed locally
- Modal CLI set up (`py -m pip install modal` then `py -m modal setup`)

Deploy the unified app (recommended):

```bash
py -m modal deploy modal_comfyui.py
```

Modal prints two endpoints:
- `...-ui.modal.run` (ComfyUI browser UI)
- `...-run.modal.run` (your SaaS runner endpoint)

Tip: keep the endpoint private and set `COMFY_RUN_TOKEN` in Modal, then send `Authorization: Bearer <token>`.

## Cost note (important)

If you leave the UI tab open, it keeps a WebSocket connection and the container may stay up (cost money).
Close the tab when you’re done. To force it down immediately:

```bash
py -m modal app stop comfyui
```

This app is CPU-only: we never request GPU resources and ComfyUI starts with `--cpu`.

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

## Legacy (older deployments)

If you previously deployed these, they still work but are no longer recommended:
- `modal_comfyui_headless.py` (app: `comfyui-headless`)
- `modal_comfyui_ui.py` (app: `comfyui-ui`)
