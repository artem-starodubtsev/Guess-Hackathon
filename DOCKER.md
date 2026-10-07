# Run Fashion Atlas in Docker

Use Docker Desktop with Linux containers (or Docker Engine on Linux). Run commands from the project folder. The image includes trained models, catalog photos and previews; no API key or runtime model downloads are needed. This packages inference, not a new training run.

## Windows: automatic GPU / CPU

```powershell
cd D:\Fashion-Retrieval-MVP
docker build -t fashion-atlas:latest .
powershell -ExecutionPolicy Bypass -File .\start-docker.ps1
```

Open http://127.0.0.1:8767. The launcher checks whether Docker can actually use CUDA, then enables GPU access or starts on CPU. Ctrl+C stops and removes its container. If your existing local server occupies the port, use `-Port 8768`. Add `-Cpu` to skip the GPU probe.

## Direct commands (Windows or Linux)

CPU, including on a computer without NVIDIA hardware:

```sh
docker run --rm --init -p 127.0.0.1:8767:8767 fashion-atlas:latest
```

GPU, when the host driver and Docker GPU support are configured:

```sh
docker run --rm --init --gpus all -p 127.0.0.1:8767:8767 fashion-atlas:latest
```

The application defaults to `FASHION_DEVICE=auto`: use available CUDA, otherwise CPU. `-e FASHION_DEVICE=cpu` forces CPU. Docker must expose the GPU with `--gpus all`; the application cannot grant itself hardware access. A direct Docker command with `--gpus all` can fail before Python starts on a host without GPU support; use the launcher for automatic fallback at that level.

Device and readiness: http://127.0.0.1:8767/api/health. Both text search and uploaded-image search use the selected device. Palette extraction and image preprocessing remain on CPU. CPU inference preserves the models and descriptors, but takes longer.

## Smaller CPU-only image

```sh
docker build --build-arg TORCH_FLAVOR=cpu -t fashion-atlas:cpu .
docker run --rm --init -p 127.0.0.1:8767:8767 fashion-atlas:cpu
```

The default CUDA-capable image also runs on CPU, but includes larger CUDA dependencies. Build for an x86-64 target; other architectures have not been validated.

## Move to another computer

```sh
docker save -o fashion-atlas.tar fashion-atlas:latest
```

Copy the archive to the other computer, then:

```sh
docker load -i fashion-atlas.tar
docker run --rm --init -p 127.0.0.1:8767:8767 fashion-atlas:latest
```

Copy `start-docker.ps1` too if you want the Windows automatic GPU probe. No training cache beyond image previews, virtual environment, annotations, API credentials or server logs are included. Rebuild after changing source, models or catalog. The container writes no uploaded photos to disk. Host port is bound to localhost for this local demo.
