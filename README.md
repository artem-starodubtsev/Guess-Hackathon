# GUESS Hackathon

Search the GUESS catalog by text or photo. Adjust **color, pattern and shape** priorities, find matching pieces, or explore the interactive cluster map.

## Run

Start Docker Desktop with Linux containers, then launch the prepared image:

```sh
docker run --rm --init -p 127.0.0.1:8767:8767 fashion-atlas
```

Open **http://localhost:8767**. Press Ctrl+C to stop. This command runs on CPU; add `--gpus all` to use an NVIDIA GPU supported by Docker.

Build the image once from the project folder (and rebuild after changes):

```sh
docker build -t fashion-atlas .
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
