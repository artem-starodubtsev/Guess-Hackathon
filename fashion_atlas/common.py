from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
ARTIFACTS = ROOT / "artifacts"
PRETRAINED = ROOT / "pretrained"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def record_metrics(stage, values):
    path = ARTIFACTS / "metrics.json"
    metrics = read_json(path) if path.exists() else {}
    metrics[stage] = values
    write_json(path, metrics)


def product_key(item):
    return item["style"], item["color"]
