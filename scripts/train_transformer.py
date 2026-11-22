import argparse
import sys
from pathlib import Path

import torch
from torch import nn

# Running a script directly puts its folder, not always the project root, on sys.path.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_loader.data_loader import get_batch, load_tokens
from src.models.transformer import Transformer


@torch.no_grad()
def estimate_loss(
    model: Transformer,
    tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    batches: int,
    device: torch.device,
) -> float:
    """Estimate average next-token loss without storing gradients."""
    was_training = model.training
    model.eval()
    losses = []

    for _ in range(batches):
        inputs, targets = get_batch(tokens, batch_size, context_length, device)
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("Expected the model to return a loss.")
        losses.append(loss.item())

    model.train(was_training)
    return sum(losses) / len(losses)


def main() -> None:
    """Load token data, train the Transformer, and save a checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, default=Path("data/tiny/train.h5"))
    parser.add_argument("--val-data", type=Path, default=Path("data/tiny/val.h5"))
    parser.add_argument("--output", type=Path, default=Path("models/tiny.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--n-embed", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        help="Required for CUDA; caps this process's PyTorch allocator fraction.",
    )
    args = parser.parse_args()

    if min(
        args.steps,
        args.batch_size,
        args.context_length,
        args.n_embed,
        args.n_head,
        args.n_blocks,
        args.eval_every,
        args.eval_batches,
    ) < 1:
        parser.error("training sizes and step counts must be positive")
    if args.n_embed % args.n_head != 0:
        parser.error("n-embed must be divisible by n-head")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA was requested, but PyTorch cannot access a CUDA device")
        if args.gpu_memory_fraction is None or not 0 < args.gpu_memory_fraction <= 1:
            parser.error("--device cuda requires --gpu-memory-fraction between 0 and 1")

        # This limits PyTorch's allocator; it does not reserve memory from other processes.
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"CUDA memory snapshot: {free_bytes / 2**30:.2f} GiB free of {total_bytes / 2**30:.2f} GiB")
        print("Use CUDA only with an approved production-safe budget.")
    elif args.gpu_memory_fraction is not None:
        parser.error("--gpu-memory-fraction is only valid with --device cuda")

    if not args.train_data.is_file() or not args.val_data.is_file():
        parser.error("Token files are missing; run scripts/prepare_tiny_data.py first.")

    train_tokens = load_tokens(args.train_data)
    val_tokens = load_tokens(args.val_data)
    device = torch.device(args.device)

    config = {
        "n_head": args.n_head,
        "n_embed": args.n_embed,
        "context_length": args.context_length,
        "vocab_size": 50_257,
        "n_blocks": args.n_blocks,
    }
    torch.manual_seed(0)
    model = Transformer(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameter_count:,} | device: {device}")

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = get_batch(
            train_tokens, args.batch_size, args.context_length, device
        )
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("Expected the model to return a loss.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            train_loss = estimate_loss(
                model, train_tokens, args.batch_size,
                args.context_length, args.eval_batches, device,
            )
            val_loss = estimate_loss(
                model, val_tokens, args.batch_size,
                args.context_length, args.eval_batches, device,
            )
            print(f"step {step:5d} | train loss {train_loss:.4f} | validation loss {val_loss:.4f}")

            # Keep the latest evaluated model and enough metadata to reconstruct it.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": config,
                    "tokenizer": {"type": "tiktoken", "name": "r50k_base"},
                    "step": step,
                },
                args.output,
            )

    print(f"Checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()