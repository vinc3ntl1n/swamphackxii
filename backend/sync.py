from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from dotenv import load_dotenv

from backend import config

ROOT = config.DATA_DIR.parent
PREFIX = config.DATA_DIR.name + "/"
SKIP = {"sessions"}
SINGLE_PART = TransferConfig(multipart_threshold=4 * 1024**3)
ENV = ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")


def client():
    load_dotenv(ROOT / ".env")
    missing = [name for name in ENV if not os.environ.get(name)]
    if missing:
        sys.exit(f"Set {', '.join(missing)} in .env")
    return boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"],
                        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                        region_name="auto",
                        config=Config(request_checksum_calculation="when_required",
                                      response_checksum_validation="when_required"))


def md5(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "md5").hexdigest()


def local() -> dict[str, Path]:
    files = {}
    for path in sorted(config.DATA_DIR.rglob("*")):
        rel = path.relative_to(config.DATA_DIR)
        if path.is_file() and path.name != ".DS_Store" and rel.parts[0] not in SKIP:
            files[PREFIX + rel.as_posix()] = path
    return files


def remote(s3) -> dict[str, str]:
    pages = s3.get_paginator("list_objects_v2").paginate(Bucket=config.R2_BUCKET, Prefix=PREFIX)
    return {obj["Key"]: obj["ETag"].strip('"') for page in pages for obj in page.get("Contents", [])
            if not obj["Key"].endswith("/")}


def push(s3) -> int:
    have, n = remote(s3), 0
    for key, path in local().items():
        if have.get(key) != md5(path):
            s3.upload_file(str(path), config.R2_BUCKET, key, Config=SINGLE_PART)
            print("up", key, flush=True)
            n += 1
    return n


def pull(s3) -> int:
    n = 0
    for key, etag in sorted(remote(s3).items()):
        path = ROOT / key
        if not path.exists() or md5(path) != etag:
            path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(config.R2_BUCKET, key, str(path))
            print("down", key, flush=True)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy data/ (except sessions/) to or from the R2 bucket.")
    ap.add_argument("direction", choices=("push", "pull"))
    args = ap.parse_args()
    n = {"push": push, "pull": pull}[args.direction](client())
    done = "uploaded" if args.direction == "push" else "downloaded"
    print(f"{n} file{'' if n == 1 else 's'} {done}.")


if __name__ == "__main__":
    main()
