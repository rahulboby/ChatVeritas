"""LLM clients used by ChatVeritas.

The module is intentionally flat, but heavyweight local-model dependencies
remain lazy.  Importing this module is therefore safe for the Groq and LM
Studio paths, which do not need Torch, PEFT, or Transformers.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from configs import settings as default_settings
from core.exceptions import ConfigurationError, ModelLoadError
from core.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_MESSAGE = (
    "You are ChatVeritas, a document-grounded AI assistant. "
    "Answer only using the supplied context. "
    "If the answer is not present, clearly state that there is insufficient information."
)
_PROVIDER_ALIASES = {"lm-studio": "lmstudio", "lm_studio": "lmstudio"}


class BaseLLM(ABC):
    """Stable generation contract shared by every ChatVeritas backend."""

    @abstractmethod
    def generate(self, prompt: str) -> tuple[str, dict[str, float | int]]:
        """Return generated text and project metrics for one assembled prompt."""

    def as_runnable(self):
        """Expose the backend to the LCEL pipeline without losing metrics."""
        from langchain_core.runnables import RunnableLambda

        def invoke_generation(prompt: str) -> dict[str, object]:
            response, metrics = self.generate(prompt)
            return {"response": response, "metrics": metrics}

        return RunnableLambda(invoke_generation, name="chatveritas_generation")


@dataclass(frozen=True)
class ProviderSettings:
    """Resolved settings for one OpenAI-compatible backend."""

    name: str
    model: str
    base_url: str
    api_key: str
    api_key_env: str | None
    response_format: dict[str, Any] | None


def normalise_backend(backend: str) -> Literal["local", "lmstudio"]:
    """Validate and normalise an Offline Chat backend name."""
    value = _PROVIDER_ALIASES.get(str(backend).strip().lower(), str(backend).strip().lower())
    if value not in {"local", "lmstudio"}:
        raise ConfigurationError(
            "Unsupported offline backend "
            f"'{backend}'. Choose either 'local' or 'lmstudio'."
        )
    return value  # type: ignore[return-value]


def resolve_llm_provider(
    config: Mapping[str, Any] | None = None,
    *,
    provider: str | None = None,
    settings_module: Any = default_settings,
) -> ProviderSettings:
    """Resolve a Groq or LM Studio OpenAI-compatible provider.

    ``config`` is supported as a migration aid for callers that previously
    passed the old JSON object.  New code should omit it and use the clear
    variables in :mod:`configs.settings` instead.
    """
    if config is not None:
        return _resolve_legacy_provider(config, provider)

    provider_name = (provider or settings_module.QUESTION_GENERATION_BACKEND).strip().lower()
    provider_name = _PROVIDER_ALIASES.get(provider_name, provider_name)

    if provider_name == "groq":
        return _provider_from_values(
            name="groq",
            model=settings_module.GROQ_MODEL_ID,
            base_url=settings_module.GROQ_BASE_URL,
            api_key_env=settings_module.GROQ_API_KEY_ENV,
            default_api_key=None,
            requires_api_key=True,
            response_format=settings_module.GROQ_QUESTION_RESPONSE_FORMAT,
        )

    if provider_name == "lmstudio":
        return _provider_from_values(
            name="lmstudio",
            model=settings_module.LMSTUDIO_MODEL_ID,
            base_url=settings_module.LMSTUDIO_BASE_URL,
            api_key_env=settings_module.LMSTUDIO_API_KEY_ENV,
            default_api_key=settings_module.LMSTUDIO_DEFAULT_API_KEY,
            requires_api_key=False,
            response_format=None,
        )

    raise ConfigurationError(
        f"Unsupported provider '{provider_name}'. Choose 'groq' or 'lmstudio'."
    )


def _resolve_legacy_provider(
    config: Mapping[str, Any], provider: str | None
) -> ProviderSettings:
    """Resolve the former JSON provider-profile format for transition users."""
    llm_config = config.get("llm")
    if not isinstance(llm_config, Mapping):
        raise ConfigurationError("Configuration must contain an 'llm' object.")

    provider_name = str(provider or llm_config.get("provider", "")).strip().lower()
    provider_name = _PROVIDER_ALIASES.get(provider_name, provider_name)
    profiles = llm_config.get("providers")
    if not provider_name or not isinstance(profiles, Mapping):
        raise ConfigurationError("Configuration must define llm.provider and llm.providers.")

    profile = profiles.get(provider_name)
    if not isinstance(profile, Mapping):
        available = ", ".join(sorted(str(name) for name in profiles)) or "none"
        raise ConfigurationError(
            f"Unsupported llm.provider '{provider_name}'. Available profiles: {available}."
        )

    return _provider_from_values(
        name=provider_name,
        model=profile.get("model"),
        base_url=profile.get("base_url"),
        api_key_env=profile.get("api_key_env"),
        default_api_key=profile.get("default_api_key"),
        requires_api_key=profile.get("requires_api_key", True),
        response_format=profile.get("response_format"),
    )


def _provider_from_values(
    *,
    name: str,
    model: Any,
    base_url: Any,
    api_key_env: Any,
    default_api_key: Any,
    requires_api_key: Any,
    response_format: Any,
) -> ProviderSettings:
    """Validate provider values and read credentials only when needed."""
    if not isinstance(model, str) or not model.strip():
        raise ConfigurationError(f"{name} model must be a non-empty string.")
    model = model.strip()
    if model.startswith("CHANGE_ME_"):
        raise ConfigurationError(
            f"Set the {name} model identifier in configs/settings.py before using it."
        )
    if not isinstance(base_url, str) or not base_url.strip():
        raise ConfigurationError(f"{name} base URL must be a non-empty string.")
    if api_key_env is not None and (
        not isinstance(api_key_env, str) or not api_key_env.strip()
    ):
        raise ConfigurationError(f"{name} API-key environment variable must be a non-empty string.")
    if not isinstance(requires_api_key, bool):
        raise ConfigurationError(f"{name} requires_api_key must be a boolean.")
    if response_format is not None and not isinstance(response_format, dict):
        raise ConfigurationError(f"{name} response_format must be a dictionary or None.")

    api_key = os.getenv(api_key_env) if api_key_env else None
    if not api_key:
        api_key = default_api_key
    if requires_api_key and (not isinstance(api_key, str) or not api_key.strip()):
        raise ConfigurationError(
            f"{api_key_env or 'An API key'} is required for the '{name}' provider. "
            "Set it in .env or in the environment and restart the application."
        )
    if not isinstance(api_key, str) or not api_key.strip():
        raise ConfigurationError(f"{name} must define an API key or a default local key.")

    return ProviderSettings(
        name=name,
        model=model,
        base_url=base_url.rstrip("/"),
        api_key=api_key.strip(),
        api_key_env=api_key_env,
        response_format=response_format,
    )


class DeployLLM(BaseLLM):
    """Cloud generation client.  Deploy mode is deliberately Groq-only."""

    def __init__(self, settings_module: Any = default_settings) -> None:
        from openai import OpenAI

        self.settings = settings_module
        self.provider = resolve_llm_provider(provider="groq", settings_module=settings_module)
        self.client = OpenAI(api_key=self.provider.api_key, base_url=self.provider.base_url)
        logger.info("DeployLLM ready | provider=groq | model=%s", self.provider.model)

    def generate(self, prompt: str) -> tuple[str, dict[str, float | int]]:
        return _generate_openai_completion(
            client=self.client,
            provider=self.provider,
            prompt=prompt,
            temperature=self.settings.GENERATION_TEMPERATURE,
            max_tokens=self.settings.GENERATION_MAX_NEW_TOKENS,
        )


class OfflineLLM(BaseLLM):
    """Offline Chat backend for either the local model or LM Studio."""

    def __init__(
        self,
        backend: str = "local",
        use_lora: bool = True,
        settings_module: Any = default_settings,
    ) -> None:
        self.settings = settings_module
        self.backend = normalise_backend(backend)
        self.use_lora = bool(use_lora) and self.backend == "local"

        if self.backend == "lmstudio":
            from openai import OpenAI

            self.provider = resolve_llm_provider(provider="lmstudio", settings_module=settings_module)
            self.client = OpenAI(api_key=self.provider.api_key, base_url=self.provider.base_url)
            logger.info("OfflineLLM ready | backend=lmstudio | model=%s", self.provider.model)
            return

        self._load_local_model()

    def generate(self, prompt: str) -> tuple[str, dict[str, float | int]]:
        if self.backend == "lmstudio":
            return _generate_openai_completion(
                client=self.client,
                provider=self.provider,
                prompt=prompt,
                temperature=self.settings.GENERATION_TEMPERATURE,
                max_tokens=self.settings.GENERATION_MAX_NEW_TOKENS,
            )
        return self._generate_local(prompt)

    def _load_local_model(self) -> None:
        """Load the local HF model only for the ``local`` backend."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        self._torch = torch
        self._peft_model = PeftModel
        self._auto_model = AutoModelForCausalLM
        self._auto_tokenizer = AutoTokenizer

        adapter_source = self._resolve_adapter() if self.use_lora else None
        tokenizer_source = self._resolve_tokenizer_source(adapter_source)
        self._load_tokenizer(tokenizer_source)
        base_model = self._load_base_model()
        self._apply_lora(base_model, adapter_source)
        self.model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "OfflineLLM ready | backend=local | base=%s | adapter=%s",
            self.settings.BASE_MODEL_ID,
            adapter_source if self.use_lora else "disabled",
        )

    def _generate_local(self, prompt: str) -> tuple[str, dict[str, float | int]]:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        temperature = self.settings.GENERATION_TEMPERATURE
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.settings.GENERATION_MAX_NEW_TOKENS,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with self._torch.inference_mode():
            started = time.perf_counter()
            outputs = self.model.generate(**inputs, **generation_kwargs)
            generation_time = time.perf_counter() - started

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip(), {
            "generation_time": generation_time,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
        }

    def _resolve_adapter(self) -> str | Path:
        adapter_path = Path(self.settings.ADAPTER_DIR)
        if self._is_complete_adapter(adapter_path):
            logger.info("Using local LoRA adapter: %s", adapter_path)
            return adapter_path
        logger.info("Local adapter incomplete at %s; using Hub adapter %s", adapter_path, self.settings.ADAPTER_REPO_ID)
        return self.settings.ADAPTER_REPO_ID

    @staticmethod
    def _is_complete_adapter(path: Path) -> bool:
        return (path / "adapter_config.json").is_file() and (
            (path / "adapter_model.safetensors").is_file()
            or (path / "adapter_model.bin").is_file()
        )

    def _resolve_tokenizer_source(self, adapter_source: str | Path | None) -> str:
        if adapter_source is None:
            return self.settings.BASE_MODEL_ID
        adapter_path = Path(str(adapter_source))
        if adapter_path.is_dir() and (adapter_path / "tokenizer_config.json").is_file():
            return str(adapter_path)
        if isinstance(adapter_source, str):
            return adapter_source
        return self.settings.BASE_MODEL_ID

    def _load_tokenizer(self, tokenizer_source: str) -> None:
        try:
            self.tokenizer = self._auto_tokenizer.from_pretrained(tokenizer_source)
        except Exception:
            logger.warning("Tokenizer load failed from %s; using base model tokenizer.", tokenizer_source)
            self.tokenizer = self._auto_tokenizer.from_pretrained(self.settings.BASE_MODEL_ID)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_base_model(self):
        torch = self._torch
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        max_memory: dict[Any, str] = {"cpu": self.settings.LOCAL_MODEL_MAX_CPU_MEMORY}
        if torch.cuda.is_available():
            max_memory[0] = self.settings.LOCAL_MODEL_MAX_GPU_MEMORY
        self.settings.MODEL_OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
        try:
            return self._auto_model.from_pretrained(
                self.settings.BASE_MODEL_ID,
                torch_dtype=dtype,
                device_map="auto",
                max_memory=max_memory,
                offload_folder=self.settings.MODEL_OFFLOAD_DIR,
                offload_state_dict=True,
                low_cpu_mem_usage=True,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load base model '{self.settings.BASE_MODEL_ID}': {exc}"
            ) from exc

    def _apply_lora(self, base_model: Any, adapter_source: str | Path | None) -> None:
        if not self.use_lora or adapter_source is None:
            self.model = base_model
            return
        try:
            self.model = self._peft_model.from_pretrained(
                base_model, str(adapter_source), is_trainable=False
            )
        except Exception as exc:
            logger.warning("LoRA adapter failed to load (%s); using the base model.", exc)
            self.model = base_model


def _generate_openai_completion(
    *,
    client: Any,
    provider: ProviderSettings,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict[str, float | int]]:
    """Generate safely from an OpenAI-compatible service."""
    started = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=provider.model,
            messages=[
                {"role": "system", "content": _SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise RuntimeError(f"{provider.name} API request failed: {exc}") from exc

    message = completion.choices[0].message if getattr(completion, "choices", None) else None
    response = getattr(message, "content", None)
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError(f"{provider.name} returned an empty response.")
    usage = getattr(completion, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
    return response.strip(), {
        "generation_time": time.perf_counter() - started,
        "prompt_tokens": int(prompt_tokens or 0),
    }
