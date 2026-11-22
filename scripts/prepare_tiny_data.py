import argparse
import json
import os
from pathlib import Path
from urllib.request import urlopen

import h5py
import numpy as np
import tiktoken


END_OF_TEXT = "<|endoftext|>"
TRAIN_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-train.txt"
)
VALIDATION_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-valid.txt"
)


def download_text(url: str, path: Path, max_bytes: int | None = None) -> Path:
    """Download and cache a text file, with an optional byte limit."""
    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(path.suffix + ".part")
    written = 0

    with urlopen(url, timeout=60) as response, partial_path.open("wb") as output:
        while max_bytes is None or written < max_bytes:
            chunk_size = 1024 * 1024
            if max_bytes is not None:
                chunk_size = min(chunk_size, max_bytes - written)
            chunk = response.read(chunk_size)
            if not chunk:
                break
            output.write(chunk)
            written += len(chunk)

    if written == 0:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded an empty file from {url}")

    # Rename only after the download finishes, so an incomplete file isn't treated as cached.
    os.replace(partial_path, path)
    return path


def split_stories(raw_text: str) -> list[str]:
    """Split on TinyStories document markers and discard a truncated final story."""
    stories = [story.strip() for story in raw_text.split(END_OF_TEXT)]
    if not raw_text.rstrip().endswith(END_OF_TEXT) and len(stories) > 1:
        stories = stories[:-1]
    return [story for story in stories if story]


def encode_stories(stories: list[str], tokenizer: tiktoken.Encoding) -> np.ndarray:
    """Encode each story and append its end-of-text token."""
    end_id = tokenizer.encode(END_OF_TEXT, allowed_special={END_OF_TEXT})[0]
    token_ids = []
    for story in stories:
        token_ids.extend(tokenizer.encode_ordinary(story))
        token_ids.append(end_id)
    return np.asarray(token_ids, dtype=np.int32)


def save_tokens(
    path: Path,
    tokens: np.ndarray,
    tokenizer_name: str,
    source: str,
) -> None:
    """Save a flat token stream and its tokenizer metadata in an HDF5 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as output:
        dataset = output.create_dataset("tokens", data=tokens)
        dataset.attrs["tokenizer"] = json.dumps(
            {"type": "tiktoken", "name": tokenizer_name}
        )
        dataset.attrs["vocab_size"] = 50_257
        dataset.attrs["source"] = source


def main() -> None:
    """Download TinyStories, tokenize it, and save train/validation files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train-mb", type=float, default=25)
    parser.add_argument("--data-dir", type=Path, default=Path("data/tiny"))
    args = parser.parse_args()

    if args.max_train_mb <= 0:
        parser.error("max-train-mb must be greater than zero")

    tokenizer_name = "r50k_base"
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    raw_dir = args.data_dir / "raw"
    train_path = raw_dir / f"train_{args.max_train_mb:g}mb.txt"
    validation_path = raw_dir / "validation.txt"

    print(f"Downloading training data (up to {args.max_train_mb:g} MB)...", flush=True)

    download_text(TRAIN_URL, train_path, int(args.max_train_mb * 1_000_000))
    print("Downloading validation data...", flush=True)

    download_text(VALIDATION_URL, validation_path)

    print("Splitting training stories...", flush=True)
    train_stories = split_stories(
        train_path.read_text(encoding="utf-8", errors="replace")
    )
    print("Splitting validation stories...", flush=True)
    validation_stories = split_stories(
        validation_path.read_text(encoding="utf-8", errors="replace")
    )
    if not train_stories or not validation_stories:
        parser.error("The downloaded files contained no complete stories")

    print("Encoding training tokens...", flush=True)
    train_tokens = encode_stories(train_stories, tokenizer)
    validation_tokens = encode_stories(validation_stories, tokenizer)
    print("saving token streams...", flush=True)
    source = "TinyStories V2 (roneneldan/TinyStories)"

    save_tokens(args.data_dir / "train.h5", train_tokens, tokenizer_name, source)
    save_tokens(args.data_dir / "val.h5", validation_tokens, tokenizer_name, source)

    print(f"Train: {len(train_tokens):,} tokens -> {args.data_dir / 'train.h5'}")
    print(f"Validation: {len(validation_tokens):,} tokens -> {args.data_dir / 'val.h5'}")

if __name__ == "__main__":
    main()