"""Download a Keras-compatible MNIST ``.npz`` file for dot-response fit.

The pipeline expects ``x_train`` and ``y_train`` arrays.  The public Keras
dataset archive already uses that layout, so no TensorFlow dependency is
needed here.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request

import numpy as np


DEFAULT_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DEFAULT_OUTPUT = "~/data/mnist.npz"


def _validate_npz(path: str) -> None:
    data = np.load(path)
    missing = {"x_train", "y_train"} - set(data.files)
    if missing:
        raise ValueError(f"downloaded file is missing arrays: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MNIST for dot-response fit")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(output), exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".mnist-", suffix=".npz", dir=os.path.dirname(output)
    )
    os.close(fd)

    try:
        print(f"Downloading: {args.url}")
        print(f"Output     : {output}")
        urllib.request.urlretrieve(args.url, tmp_path)
        _validate_npz(tmp_path)
        os.replace(tmp_path, output)
    except Exception as exc:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        print(f"ERROR: MNIST download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("MNIST ready.")


if __name__ == "__main__":
    main()
