# Docker

Use Docker Desktop with Linux containers or Docker Engine on Linux.

## Start

```sh
docker run --rm --init -p 127.0.0.1:8767:8767 fashion-atlas
```

Open http://localhost:8767. Ctrl+C stops and removes the container. Use `8768:8767` instead to change the host port.

## Build

From the project folder, with the local catalog and model files present:

```sh
docker build -t fashion-atlas .
```

Rebuild after changing code, models or catalog. The image includes everything needed for inference; no API key, downloads or host data mounts are needed when running it. GitHub contains the code only: supply `data/`, `pretrained/`, `artifacts/` and `cache/images.npz` before building.

## GPU (optional)

```sh
docker run --rm --init --gpus all -p 127.0.0.1:8767:8767 fashion-atlas
```

The default command works on CPU. With `--gpus all`, Docker exposes the NVIDIA GPU and the application automatically uses CUDA when available. If Docker has no GPU support, omit that flag: Docker can reject it before the application starts. `-e FASHION_DEVICE=cpu` forces CPU. Check the active device at http://localhost:8767/api/health.

For a smaller CPU-only image: `docker build --build-arg TORCH_FLAVOR=cpu -t fashion-atlas .`.

## Another computer

Export with `docker save -o fashion-atlas.tar fashion-atlas`, copy the archive, then import with `docker load -i fashion-atlas.tar`. Start with the same one-line command above. Validated on x86-64.
