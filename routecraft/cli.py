from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from .data import BASE_ID, BASE_REVISION, DATA_COMMIT, load, prepare

SOURCE_SHA = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
BASE_HASHES = {
    "config.json": "508e1f01aae55d73355cbdd82609be2f43ba5a0d3428837adbe56cf8391f8b39",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "model.safetensors": "7fb69ad9f6866d8983183c930e33828f326470bf6ad8bbb2ad4ed957a92e9414",
}


def download(url, path, expected):
    path = Path(path)
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Source checksum mismatch for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    temporary.write_bytes(raw)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="RouteCraft training, verification and serving")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8030)
    verify = sub.add_parser("verify")
    verify.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    acquire = sub.add_parser("prepare")
    acquire.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    acquire.add_argument("--data", type=Path, default=Path("data/task.json"))
    train = sub.add_parser("train")
    train.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    train.add_argument("--data", type=Path, default=Path("data/task.json"))
    train.add_argument("--epochs", type=int, default=24)
    train.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run("routecraft.api:create_app", factory=True, host=args.host, port=args.port)
    elif args.command == "verify":
        from .engine import verify_bundle

        print(json.dumps(verify_bundle(args.artifacts), indent=2))
    elif args.command == "prepare":
        for filename, checksum in BASE_HASHES.items():
            download(
                f"https://huggingface.co/{BASE_ID}/resolve/{BASE_REVISION}/{filename}",
                args.artifacts / "base" / filename,
                checksum,
            )
        source = Path("runtime/source_full.json")
        download(
            f"https://raw.githubusercontent.com/clinc/oos-eval/{DATA_COMMIT}/data/data_full.json",
            source,
            SOURCE_SHA,
        )
        data = prepare(source, args.data)
        print(json.dumps({k: len(v) for k, v in data["splits"].items()}))
    else:
        from .train import train as run

        load(args.data)
        report = run(args.data, args.artifacts, epochs=args.epochs, seed=args.seed)
        print(json.dumps({k: v["macro_f1"] for k, v in report["test"].items()}, indent=2))


if __name__ == "__main__":
    main()
