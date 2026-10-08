import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM

from kvreuse.forward import ModelView


@pytest.fixture(scope="session")
def view():
    """Tiny random Mistral in fp32 on CPU. Same architecture and RoPE as Mistral-7B v0.3
    (theta 1e6, GQA, no sliding window), so E0 logic is exercised without a GPU."""
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = MistralConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1024,
        rope_theta=1e6,
        sliding_window=None,
        initializer_range=0.2,  # large enough that attention is far from uniform
    )
    model = MistralForCausalLM(cfg).eval().to(torch.float32)
    model.config._attn_implementation = "sdpa"
    return ModelView(model)


@pytest.fixture
def ids():
    g = torch.Generator().manual_seed(1)
    return lambda n: torch.randint(5, 256, (n,), generator=g).tolist()
