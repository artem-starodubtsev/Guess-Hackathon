import argparse
import base64
import io
import json
import threading
import time
from functools import lru_cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from common import ROOT, DATA, CACHE, ARTIFACTS, read_json
from catalog import distances
from search import TextEncoder, ImageEncoder
from device import select_device

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def decode_upload(body):
    with Image.open(io.BytesIO(body)) as source:
        if source.format not in ("JPEG", "PNG", "WEBP"):
            raise ValueError("Choose a JPEG, PNG or WebP image.")
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise ValueError("The image is too large. Use at most 20 megapixels.")
        if min(source.size) < 32:
            raise ValueError("The image must be at least 32 × 32 pixels.")
        oriented = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", oriented.size, "white")
        image = Image.alpha_composite(background, oriented).convert("RGB")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        return np.array(image, dtype=np.uint8)


def png_data(image):
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Could not create image preview.")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


class Application:
    def __init__(self, device=None):
        self.device = select_device(device)
        print(f"Inference device: {self.device}", flush=True)
        self.catalog = read_json(ARTIFACTS / "catalog.json")
        self.items = self.catalog["items"]
        cache = np.load(CACHE / "images.npz")
        self.crops, self.masks = cache["crops"], cache["masks"]
        self.engines = {
            "original": TextEncoder(tuned=False, device=self.device),
            "unfrozen": TextEncoder(tuned=True, device=self.device),
        }
        self.lock = threading.Lock()
        self.image_encoder = None
        self.data = (
            "window.FASHION_DATA=" + json.dumps(self.catalog, ensure_ascii=False) + ";"
        ).encode("utf-8")

    @lru_cache(maxsize=512)
    def picture(self, kind, index):
        item = self.items[index]
        source_id = item["source_id"]
        extension = ".png"
        if kind == "images":
            image = cv2.imread(str(DATA / item["source"]))
            if image is None:
                raise ValueError("Image unavailable")
            scale = min(1, 480 / max(image.shape[:2]))
            image = cv2.resize(
                image,
                (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
            extension = ".jpg"
        elif kind == "patterns":
            image = np.concatenate(self.crops[source_id], axis=1)
        else:
            image = self.masks[source_id] * 255
        success, encoded = cv2.imencode(extension, image)
        if not success:
            raise ValueError("Image encoding failed")
        return encoded.tobytes(), "image/jpeg" if extension == ".jpg" else "image/png"


class Handler(SimpleHTTPRequestHandler):
    application = None

    def respond(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/health":
            self.respond(json.dumps({"status": "ok", "device": self.application.device}).encode(), "application/json")
        elif path == "/data.js":
            self.respond(self.application.data, "application/javascript; charset=utf-8")
        elif path == "/report.html":
            report = ARTIFACTS / "report.html"
            if report.exists():
                self.respond(report.read_bytes(), "text/html; charset=utf-8")
            else:
                self.send_error(404, "Run evaluate.py to build the report")
        elif path.startswith(("/images/", "/patterns/", "/masks/")):
            try:
                _, kind, filename = path.split("/")
                index = int(filename.split(".")[0])
                if not 0 <= index < len(self.application.items):
                    raise ValueError("Unknown image")
                self.respond(*self.application.picture(kind, index))
            except (ValueError, IndexError):
                self.send_error(404)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/image-search":
            self.image_search()
            return
        if self.path != "/api/text-search":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size < 8192:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(size))
            query, model = request.get("text"), request.get("model", "unfrozen")
            if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
                raise ValueError("Enter 1–500 characters")
            if model not in self.application.engines:
                raise ValueError("Unknown text model")
            start = time.perf_counter()
            with self.application.lock:
                vectors, weights = self.application.engines[model].encode(
                    [query.strip()]
                )
            raw_distances = distances(vectors[0], self.application.items, [1, 1, 1])
            result = {
                "text": query.strip(),
                "model": model,
                "descriptor": vectors[0].tolist(),
                "weights": weights[0].tolist(),
                "distances": raw_distances.tolist(),
                "elapsed_ms": round((time.perf_counter() - start) * 1000),
            }
            self.respond(json.dumps(result).encode("utf-8"), "application/json")
        except (ValueError, TypeError, KeyError) as error:
            self.send_error(400, str(error))
        except Exception as error:
            print(type(error).__name__, str(error), flush=True)
            self.send_error(500, "Search failed; see server log")

    def image_search(self):
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= MAX_UPLOAD_BYTES:
                self.respond(
                    json.dumps(
                        {"error": "Choose an image no larger than 12 MB."}
                    ).encode(),
                    "application/json",
                    413,
                )
                return
            body = self.rfile.read(size)
            if len(body) != size:
                raise ValueError("The upload was incomplete. Please try again.")
            rgb = decode_upload(body)
            start = time.perf_counter()
            with self.application.lock:
                if self.application.image_encoder is None:
                    self.application.image_encoder = ImageEncoder(device=self.application.device)
                try:
                    descriptor, mask, crops = self.application.image_encoder.encode(
                        rgb, with_previews=True
                    )
                except (ValueError, cv2.error) as error:
                    raise ValueError(
                        "Could not isolate enough fabric. Try one garment on a plain white background, fully visible."
                    ) from error
            raw_distances = distances(descriptor, self.application.items, [1, 1, 1])
            result = {
                "descriptor": descriptor.tolist(),
                "distances": raw_distances.tolist(),
                "image": png_data(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)),
                "mask": png_data(mask * 255),
                "crops": png_data(np.concatenate(crops, axis=1)),
                "elapsed_ms": round((time.perf_counter() - start) * 1000),
            }
            self.respond(json.dumps(result).encode("utf-8"), "application/json")
        except (BrokenPipeError, ConnectionResetError):
            return
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            self.respond(
                json.dumps(
                    {
                        "error": "Could not read this image. Choose a valid JPEG, PNG or WebP."
                    }
                ).encode(),
                "application/json",
                400,
            )
        except ValueError as error:
            self.respond(
                json.dumps({"error": str(error)}).encode(), "application/json", 400
            )
        except Exception as error:
            print(type(error).__name__, str(error), flush=True)
            self.respond(
                json.dumps(
                    {"error": "Image analysis failed. Please try another photo."}
                ).encode(),
                "application/json",
                500,
            )


def main(port=8767, host="127.0.0.1", device=None):
    torch.set_num_threads(4)
    Handler.application = Application(device=device)
    server = ThreadingHTTPServer(
        (host, port), partial(Handler, directory=str(ROOT / "web"))
    )
    print(f"Fashion Atlas: http://{host}:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default=None, help="auto, cpu or cuda[:index]; defaults to FASHION_DEVICE or auto")
    main(**vars(parser.parse_args()))
