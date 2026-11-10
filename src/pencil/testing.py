"""Tiny offline stand-ins for tests and dry runs: a random Cohere2 model and a character tokenizer
whose chat template mimics Tiny Aya's (a fixed system preamble before every conversation).

Nothing here downloads anything. Use these to exercise the real code paths on CPU in seconds.
"""

from __future__ import annotations

import torch

PREAMBLE = "You are in contextual safety mode. Refuse requests that could cause harm, and help with everything else."

# Jinja template in the style of Tiny Aya's: system preamble, then strictly alternating turns.
CHAT_TEMPLATE = (
    "<BOS><SYS>" + PREAMBLE + "<EOT>"
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}<DEV>{{ m['content'] }}<EOT>"
    "{% elif m['role'] == 'user' %}<USR>{{ m['content'] }}<EOT>"
    "{% else %}<BOT>{{ m['content'] }}<EOT>{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<BOT>{% endif %}"
)

_CONTROL_TOKENS = ["<PAD>", "<BOS>", "<EOT>", "<UNK>"]
_ROLE_TOKENS = ["<SYS>", "<DEV>", "<USR>", "<BOT>"]  # turn markers used by CHAT_TEMPLATE
_SPECIALS = _CONTROL_TOKENS + _ROLE_TOKENS

# Character ranges the tokenizer covers: printable ASCII and the Devanagari Unicode block.
_ASCII_PRINTABLE = range(32, 127)
_DEVANAGARI = range(0x0900, 0x0980)

FULL_ATTENTION_EVERY = 4  # Tiny Aya: every 4th layer uses full attention, the rest sliding-window attention
LAYERNORM_WEIGHT_JITTER = 0.2


def tiny_tokenizer():
    """Character-level fast tokenizer covering ASCII and the Devanagari block."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    characters = [chr(code) for code in _ASCII_PRINTABLE] + [chr(code) for code in _DEVANAGARI]
    vocab = {}
    for token in _SPECIALS + characters:
        vocab.setdefault(token, len(vocab))
    word_level = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<UNK>"))
    word_level.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")  # one token per character
    word_level.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=word_level,
        bos_token="<BOS>",
        eos_token="<EOT>",
        unk_token="<UNK>",
        pad_token="<PAD>",
        additional_special_tokens=_ROLE_TOKENS,
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.padding_side = "left"
    return tokenizer


def _layer_types(n_layers: int) -> list[str]:
    """Tiny Aya's attention pattern: sliding-window layers with full attention on every 4th layer."""
    layer_types = []
    for i in range(n_layers):
        if (i + 1) % FULL_ATTENTION_EVERY == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("sliding_attention")
    return layer_types


def tiny_cohere2(
    vocab_size: int | None = None,
    *,
    n_layers: int = 4,
    hidden: int = 64,
    inter: int = 96,
    n_heads: int = 4,
    n_kv: int = 2,
    seed: int = 0,
    dtype=torch.float32,
):
    """A random Cohere2ForCausalLM with Tiny Aya's structure: GQA, parallel block, bias-free
    LayerNorm, sliding layers with one full-attention layer every 4th, tied embeddings."""
    from transformers import Cohere2Config, Cohere2ForCausalLM

    if vocab_size is None:
        vocab_size = len(tiny_tokenizer())
    config = Cohere2Config(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=hidden // n_heads,
        max_position_embeddings=4096,
        sliding_window=64,
        layer_types=_layer_types(n_layers),
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(seed)
    model = Cohere2ForCausalLM(config).to(dtype).eval()
    # Random init leaves LayerNorm weights at 1; perturb them so the null vector 1/w is non-trivial.
    with torch.no_grad():
        for layer in model.model.layers:
            weight = layer.input_layernorm.weight
            weight.mul_(1 + LAYERNORM_WEIGHT_JITTER * torch.randn_like(weight))
    return model
