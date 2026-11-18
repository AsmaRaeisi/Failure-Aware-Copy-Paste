"""Deterministic random streams for paired ablation datasets."""

import hashlib
import random


def stable_seed(master_seed, image_key, stream_name):
    """Return a stable 64-bit seed for one image/component stream.

    Python's built-in hash() is intentionally randomized between processes, so
    it must not be used for reproducible experiment seeds.
    """
    text = f"{int(master_seed)}|{image_key}|{stream_name}".encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def make_rng(master_seed, image_key, stream_name):
    """Create an independent RNG for one image and one augmentation component."""
    return random.Random(stable_seed(master_seed, image_key, stream_name))
