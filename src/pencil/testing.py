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

_SPECIALS = ["<PAD>", "<BOS>", "<EOT>", "<UNK>", "<SYS>", "<DEV>", "<USR>", "<BOT>"]


def tiny_tokenizer():
    """Character-level fast tokenizer covering ASCII and the Devanagari block."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    chars = [chr(c) for c in range(32, 127)] + [chr(c) for c in range(0x0900, 0x0980)]
    vocab = {t: i for i, t in enumerate(_SPECIALS)}
    for ch in chars:
        vocab.setdefault(ch, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<UNK>"))
    tk.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    tk.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        bos_token="<BOS>",
        eos_token="<EOT>",
        unk_token="<UNK>",
        pad_token="<PAD>",
        additional_special_tokens=_SPECIALS[4:],
    )
    tok.chat_template = CHAT_TEMPLATE
    tok.padding_side = "left"
    return tok


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
    layer_types = ["full_attention" if (i + 1) % 4 == 0 else "sliding_attention" for i in range(n_layers)]
    cfg = Cohere2Config(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=hidden // n_heads,
        max_position_embeddings=4096,
        sliding_window=64,
        layer_types=layer_types,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(seed)
    model = Cohere2ForCausalLM(cfg).to(dtype).eval()
    # Random init leaves LayerNorm weights at 1; perturb them so the null vector 1/w is non-trivial.
    with torch.no_grad():
        for layer in model.model.layers:
            layer.input_layernorm.weight.mul_(1 + 0.2 * torch.randn_like(layer.input_layernorm.weight))
    return model
