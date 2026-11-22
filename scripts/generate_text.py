import argparse
import sys
from pathlib import Path

import tiktoken
import torch

# Running a script directly puts its folder, not always the project root, on sys.path.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.transformer import Transformer


def main() -> None:
    """Load a checkpoint and print a generated text continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=Path("models/tiny.pt"))
    parser.add_argument("--input_text", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("max_new_tokens must be zero or greater")
    if args.temperature <= 0:
        parser.error("temperature must be greater than zero")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but PyTorch cannot access a CUDA device")
    if not args.model_path.is_file():
        parser.error(f"Checkpoint not found: {args.model_path}")

    device = torch.device(args.device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)

    tokenizer_info = checkpoint.get("tokenizer", {})
    if tokenizer_info.get("name") != "r50k_base":
        parser.error("This script expects a checkpoint using the r50k_base tokenizer")

    model = Transformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = tiktoken.get_encoding("r50k_base")
    prompt_tokens = tokenizer.encode_ordinary(args.input_text)
    if not prompt_tokens:
        parser.error("input_text must contain at least one token")

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    generated = output_ids[0].tolist()
    eot_id = tokenizer.encode(
    "<|endoftext|>",
    allowed_special={"<|endoftext|>"},
)[0]

# Don't display later tokens after the model marks the story as finished.
    if eot_id in generated:
        generated = generated[:generated.index(eot_id)]

    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()