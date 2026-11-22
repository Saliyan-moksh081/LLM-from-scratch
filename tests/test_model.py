import torch

from src.models.attention import Head
from src.models.transformer import Transformer


def test_transformer_returns_logits_and_loss() -> None:
    """Check output shape, scalar loss, and gradient flow."""
    model = Transformer(
        n_head=2,
        n_embed=16,
        context_length=8,
        vocab_size=32,
        n_blocks=2,
    )
    tokens = torch.randint(0, 32, (2, 8))

    logits, loss = model(tokens, targets=tokens)

    assert logits.shape == (2, 8, 32)
    assert loss is not None and loss.ndim == 0
    loss.backward()
    assert model.token_embed.weight.grad is not None


def test_attention_does_not_use_future_tokens() -> None:
    """Check that later tokens cannot affect the first position's attention output."""
    torch.manual_seed(0)
    attention = Head(head_size=4, n_embed=4, context_length=4)
    tokens = torch.randn(1, 4, 4)
    changed_future_tokens = tokens.clone()
    changed_future_tokens[:, 1:, :] += 10

    original_output = attention(tokens)
    changed_output = attention(changed_future_tokens)

    torch.testing.assert_close(original_output[:, 0], changed_output[:, 0])


def test_generate_appends_requested_number_of_tokens() -> None:
    """Check generation returns the prompt plus the requested continuation length."""
    model = Transformer(
        n_head=2,
        n_embed=16,
        context_length=8,
        vocab_size=32,
        n_blocks=2,
    )
    prompt = torch.tensor([[1, 2]])

    generated = model.generate(prompt, max_new_tokens=3)

    assert generated.shape == (1, 5)