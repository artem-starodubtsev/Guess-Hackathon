# Docker

Use Docker Desktop with Linux containers or Docker Engine on Linux (x86-64).

## Build and start

```sh
git clone https://github.com/artem-starodubtsev/Guess-Hackathon.git
cd Guess-Hackathon
docker build -t fashion-atlas .
docker run --rm --init -p 127.0.0.1:8767:8767 fashion-atlas
```

Open http://localhost:8767. Ctrl+C stops and removes the container. Change the first port to use another host port, for example `127.0.0.1:8768:8767`.

The build downloads a versioned runtime bundle from GitHub Releases and verifies the SHA-256 hash in `runtime-assets.json`. It includes catalog photos, descriptors, previews, trained heads, DINOv2 Small and the FashionCLIP text encoder. The download is cached by Docker. No access token, API key or local data folder is required. Startup and search work offline after building.

The bundle is for inference, not full retraining: synthetic training queries, source spreadsheets, training logs and disposable training caches are not included. Pretrained model licenses and source references are preserved under `pretrained/` in the bundle. Product images remain GUESS imagery.

## GPU (optional)

```sh
docker run --rm --init --gpus all -p 127.0.0.1:8767:8767 fashion-atlas
```

Without GPU exposure the application selects CPU. With `--gpus all`, it uses CUDA when available. If Docker has no GPU support, omit the flag; Docker may reject it before Python starts. Force CPU with `-e FASHION_DEVICE=cpu`. Check http://localhost:8767/api/health for the active device.

For a smaller CPU-only image, build with `docker build --build-arg TORCH_FLAVOR=cpu -t fashion-atlas .`.

## Local Python

Install `requirements.txt` in a Python 3.12 environment, then run:

```sh
python scripts/download_assets.py
python -m fashion_atlas
```

The downloader restores the pinned demo files into the project folder, replacing files with the same names. Save any custom-trained models first. Full retraining additionally requires the original training manifests and annotations described in [development.md](development.md).
