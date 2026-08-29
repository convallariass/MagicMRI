#!/usr/bin/env python3
"""Anonymously download and verify the immutable MagicMRI release checkpoint."""

import argparse
import hashlib
import os
from pathlib import Path

import gdown


FILE_ID = "1FAuVQjvqwGI6r9oGWeFFPFaAVqQS2ktM"
EXPECTED_SIZE = 1_483_011_953
EXPECTED_SHA256 = "7afcc73b8c829b96cb9276d1a7cc234a30d3182f6594f744083556db6f07e65e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path) -> None:
    size = path.stat().st_size
    if size != EXPECTED_SIZE:
        raise RuntimeError(f"Checkpoint size mismatch: expected {EXPECTED_SIZE}, got {size}")
    observed = sha256(path)
    if observed != EXPECTED_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch: expected {EXPECTED_SHA256}, got {observed}"
        )


def main():
    parser = argparse.ArgumentParser(description="Download and verify the MagicMRI checkpoint")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/magicmri_ckpt_release.pth"),
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        verify(output)
        print(f"Already present and verified: {output}")
        return
    partial = output.with_name(output.name + ".part")
    resumable = list(output.parent.glob(partial.name + "*"))
    action = "Resuming" if resumable else "Starting"
    print(f"{action} anonymous checkpoint download (Google Drive file {FILE_ID})", flush=True)
    downloaded = gdown.download(id=FILE_ID, output=str(partial), quiet=False, resume=True)
    if downloaded is None or not partial.is_file():
        raise RuntimeError("Anonymous checkpoint download did not produce a file")
    try:
        verify(partial)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, output)
    print(f"Verified size={EXPECTED_SIZE} sha256={EXPECTED_SHA256}")
    print(f"checkpoint={output}")


if __name__ == "__main__":
    main()
