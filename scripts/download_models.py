"""Download the models needed to run transcription locally, offline.

    python scripts/download_models.py            # the transcriber
    python scripts/download_models.py --punct    # and the punctuation model
    python scripts/download_models.py --list     # what is already here

Everything lands in `data/models/`, which is gitignored. The download is a few
hundred megabytes and only has to happen once.
"""

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

MODELS = {
    "asr": {
        "name": "sherpa-onnx-streaming-paraformer-trilingual-zh-cantonese-en",
        "url": f"{RELEASE}/asr-models/"
               "sherpa-onnx-streaming-paraformer-trilingual-zh-cantonese-en.tar.bz2",
        "what": "streaming Mandarin + Cantonese + English transcriber",
        # The float32 encoder/decoder double the size for no benefit here; the
        # int8 pair is what the app loads.
        "keep": ("tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx",
                 "README.md", "test_wavs"),
        "required": ("tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx"),
        "dest": lambda: config.SHERPA_MODEL_DIR,
    },
    "punct": {
        "name": "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12",
        "url": f"{RELEASE}/punctuation-models/"
               "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2",
        "what": "adds punctuation, which the transcriber does not produce",
        "keep": ("model.onnx", "tokens.json", "README.md"),
        "required": ("model.onnx",),
        "dest": lambda: config.SHERPA_PUNCTUATION_DIR,
    },
}


def already_there(spec) -> bool:
    dest = spec["dest"]()
    return all((dest / name).exists() for name in spec["required"])


def human(n: int) -> str:
    return f"{n / 1e6:.0f} MB" if n else "unknown size"


def download(spec) -> None:
    dest = spec["dest"]()
    if already_there(spec):
        print(f"  already present: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {spec['what']}")
    print(f"  from {spec['url']}")

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "model.tar.bz2"

        def progress(block, block_size, total):
            done = block * block_size
            if total > 0:
                pct = min(100, done * 100 // total)
                print(f"\r  downloading… {pct}% of {human(total)}", end="", flush=True)

        urllib.request.urlretrieve(spec["url"], archive, reporthook=progress)
        print("\r  downloading… done" + " " * 20)

        print("  extracting…", flush=True)
        with tarfile.open(archive, "r:bz2") as tar:
            wanted = [
                m for m in tar.getmembers()
                if any(f"/{k}" in m.name or m.name.endswith(f"/{k}") for k in spec["keep"])
            ]
            tar.extractall(tmp, members=wanted, filter="data")

        extracted = Path(tmp) / spec["name"]
        if not extracted.exists():
            raise SystemExit(f"  unexpected archive layout: no {spec['name']}/ inside")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(extracted), str(dest))

    missing = [n for n in spec["required"] if not (dest / n).exists()]
    if missing:
        raise SystemExit(f"  download finished but {', '.join(missing)} is missing")
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"  ready: {dest}  ({human(size)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--punct", action="store_true",
                        help="also download the punctuation model")
    parser.add_argument("--list", action="store_true",
                        help="show what is already downloaded and exit")
    args = parser.parse_args()

    if args.list:
        for key, spec in MODELS.items():
            mark = "yes" if already_there(spec) else "no "
            print(f"  [{mark}] {key:6} {spec['dest']()}")
        return

    wanted = ["asr"] + (["punct"] if args.punct else [])
    for key in wanted:
        print(f"\n=== {key} ===")
        download(MODELS[key])

    print("\nDone. Set STT_PROVIDER=local in .env to use it.")
    if not args.punct and not already_there(MODELS["punct"]):
        print("Tip: --punct also fetches the punctuation model (~280 MB), which "
              "makes the transcript much easier to read.")


if __name__ == "__main__":
    main()
