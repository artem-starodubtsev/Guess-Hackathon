# GUESS Hackathon

Search the GUESS catalog by text or photo. Adjust **color, pattern and shape** priorities, find matching pieces, or explore the interactive cluster map.

## Run

Start Docker Desktop with Linux containers. From this folder, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start-docker.ps1
```

Open **http://localhost:8767**. The launcher builds the image and uses GPU when available, otherwise CPU. Press Ctrl+C to stop. Use `-Cpu` to force CPU or `-Port 8768` to change the port.

On Linux, or to start directly on CPU:

```sh
docker build -t fashion-atlas .
docker run --rm --init -p 127.0.0.1:8767:8767 fashion-atlas
```

**Data required:** the GitHub repository contains code, not catalog photos or model weights. Before building, copy the prepared `data/`, `pretrained/`, `artifacts/` and `cache/images.npz` from the local project. No API key is needed. [Docker details](docs/docker.md).

## How it works

Each item has three visual signatures: palette (16 values), pattern (64) and shape (64). A text encoder maps requests into the same spaces. Search combines their distances using your slider weights; zero ignores an attribute.

## Files

- `fashion_atlas/` — server, models and search.
- `fashion_atlas/preprocessing/` — image, palette and silhouette processing.
- `fashion_atlas/training/` — preparation, training and evaluation.
- `web/` — interface and cluster maps.
- `docs/` — [development and training](docs/development.md).

Local launch: `python -m fashion_atlas` (after installing `requirements.txt`).
