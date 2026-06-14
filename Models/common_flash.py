import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def default_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_flash_model(
    model_id: str,
    dtype: torch.dtype | None = None,
    use_flash_attention: bool = True,
    attention_backend: str | None = None,
    device_map: str | None = "auto",
    quantization_config=None,
):
    dtype = dtype or default_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model_kwargs = {
        "torch_dtype": dtype,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    model_kwargs["attn_implementation"] = attention_backend or ("flash_attention_2" if use_flash_attention else "eager")
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    return tokenizer, model


def print_attention_config(model) -> None:
    config = model.config
    hidden_size = getattr(config, "hidden_size", None)
    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)

    print("Configuracion de atencion:")
    print(f"hidden_size: {hidden_size if hidden_size is not None else 'n/a'}")
    print(f"num_attention_heads: {num_heads if num_heads is not None else 'n/a'}")
    print(f"num_key_value_heads: {num_kv_heads if num_kv_heads is not None else 'n/a'}")

    if hidden_size is not None and num_heads:
        head_dim = hidden_size // num_heads
        exact = hidden_size % num_heads == 0
        suffix = "" if exact else " (division no exacta)"
        print(f"head_dim: {head_dim}{suffix}")
    else:
        print("head_dim: n/a")


def add_common_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-id", required=True, help="Hugging Face model id or local path.")
    return parser


def print_loaded(model_id: str, model, use_flash_attention: bool = True, attention_backend: str | None = None) -> None:
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    print(f"Loaded {model_id}")
    print(f"Attention: {attention_backend or ('flash_attention_2' if use_flash_attention else 'eager')}")
    print(f"Dtype: {dtype}")
    print(f"Device: {device}")
