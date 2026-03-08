from __future__ import annotations

import time
from typing import Any

from .llm_clients import ChatResult, LLMError
from .runtime_config import LLMConfig


class TransformersChatClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None

    def _load_components(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = (
                self._import_transformers_modules()
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                "Transformers backend requires 'transformers' and 'torch' packages"
            ) from exc

        quantization_config = None
        if self.config.load_in_4bit:
            if BitsAndBytesConfig is None:
                raise LLMError(
                    "load_in_4bit=true requires bitsandbytes support in transformers"
                )
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        model_kwargs: dict[str, Any] = {
            "device_map": self.config.device_map,
        }
        if self.config.torch_dtype != "auto":
            if not hasattr(torch, self.config.torch_dtype):
                raise LLMError(f"Unsupported torch_dtype: {self.config.torch_dtype}")
            model_kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self._model = AutoModelForCausalLM.from_pretrained(self.config.model, **model_kwargs)
        return self._tokenizer, self._model

    def _import_transformers_modules(self) -> tuple[Any, Any, Any, Any | None]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            from transformers import BitsAndBytesConfig
        except Exception:  # noqa: BLE001
            BitsAndBytesConfig = None
        return torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    def chat(self, messages: list[dict[str, str]], timeout_seconds: float) -> ChatResult:
        del timeout_seconds  # HF local generation path does not use request timeout.
        tokenizer, model = self._load_components()

        if hasattr(tokenizer, "apply_chat_template"):
            chat_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            chat_text = "\n".join(
                f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages
            )
            chat_text += "\nassistant:"
        inputs = tokenizer(chat_text, return_tensors="pt")
        model_device = getattr(model, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
        }
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            generate_kwargs["pad_token_id"] = eos_token_id
        if self.config.temperature <= 0:
            generate_kwargs["do_sample"] = False
        else:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.config.temperature
            generate_kwargs["top_p"] = self.config.top_p

        start = time.perf_counter()
        output_ids = model.generate(**inputs, **generate_kwargs)
        latency_ms = int((time.perf_counter() - start) * 1000)
        input_len = inputs["input_ids"].shape[-1]
        content = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

        return ChatResult(
            text=content,
            latency_ms=latency_ms,
            raw={
                "backend": "transformers",
                "model": self.config.model,
                "max_new_tokens": self.config.max_new_tokens,
                "load_in_4bit": self.config.load_in_4bit,
            },
        )
