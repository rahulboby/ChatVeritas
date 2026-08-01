"""
llm/offline.py

Offline LLM backend for ChatVeritas.

Loads the Qwen 2.5 (3B) base model and optionally applies a LoRA
fine-tuning adapter via PEFT.

Adapter resolution order
------------------------
1. Local filesystem path (``config["model"]["adapter_path"]``).
   Checked for a complete adapter (``adapter_config.json`` +
   ``adapter_model.safetensors`` / ``adapter_model.bin``).
2. Hugging Face Hub repository ID (``config["model"]["adapter_repo_id"]``).
   Used as a fallback if the local adapter is absent or incomplete.

This allows the same code to work both:
- locally (uses the locally fine-tuned checkpoint), and
- via Streamlit Cloud (downloads from the Hub on first run).
"""

import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

from core.exceptions import ModelLoadError
from core.logger import get_logger
from llm.base import BaseLLM

hf_logging.set_verbosity_error()

logger = get_logger(__name__)

# Project root: llm/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class OfflineLLM(BaseLLM):
    """
    Local inference backend using Qwen 2.5 (3B) with optional LoRA.

    Parameters
    ----------
    config : dict
        Loaded application configuration dictionary.
    use_lora : bool
        Whether to apply the LoRA fine-tuning adapter.
    """

    def __init__(self, config: dict, use_lora: bool = True) -> None:
        self.config = config
        self.use_lora = use_lora
        self._load()

    # ------------------------------------------------------------------
    # BaseLLM implementation
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> tuple[str, dict]:
        """
        Generate a response using the local Qwen model.

        Parameters
        ----------
        prompt : str
            The assembled RAG prompt.

        Returns
        -------
        tuple[str, dict]
            ``(response_text, {"generation_time": float, "prompt_tokens": int})``
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        temperature = self.config["generation"]["temperature"]
        generation_kwargs: dict = {
            "max_new_tokens": self.config["generation"]["max_new_tokens"],
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        logger.debug(
            "Generating response | max_new_tokens=%d | temperature=%.2f",
            generation_kwargs["max_new_tokens"],
            temperature,
        )

        with torch.inference_mode():
            gen_start = time.perf_counter()
            outputs = self.model.generate(**inputs, **generation_kwargs)
            generation_time = time.perf_counter() - gen_start

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        metrics = {
            "generation_time": generation_time,
            "prompt_tokens": inputs["input_ids"].shape[1],
        }

        logger.debug(
            "Generation complete | %.2f s | %d prompt tokens",
            generation_time,
            metrics["prompt_tokens"],
        )

        return response.strip(), metrics

    # ------------------------------------------------------------------
    # Private loading helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Orchestrate tokenizer, base model, and optional LoRA loading."""
        logger.info(
            "Loading OfflineLLM | base=%s | use_lora=%s",
            self.config["model"]["base_model"],
            self.use_lora,
        )

        adapter_source = self._resolve_adapter() if self.use_lora else None
        tokenizer_source = self._resolve_tokenizer_source(adapter_source)

        self._load_tokenizer(tokenizer_source)
        base_model = self._load_base_model()
        self._apply_lora(base_model, adapter_source)
        self.model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared after model load.")

        logger.info(
            "OfflineLLM ready | backend=%s | adapter=%s",
            self.config["model"]["base_model"],
            adapter_source if self.use_lora else "disabled",
        )

    def _resolve_adapter(self) -> str | Path:
        """
        Resolve the adapter location.

        Returns
        -------
        str | Path
            Local ``Path`` if a complete adapter exists there, otherwise
            the HF Hub repo ID string.
        """
        local_path = Path(self.config["model"]["adapter_path"])
        if not local_path.is_absolute():
            local_path = _PROJECT_ROOT / local_path

        if self._is_complete_adapter(local_path):
            logger.info("Using local LoRA adapter: %s", local_path)
            return local_path

        hub_id: str = self.config["model"]["adapter_repo_id"]
        logger.info(
            "Local adapter incomplete at %s → falling back to HF Hub: %s",
            local_path,
            hub_id,
        )
        return hub_id

    @staticmethod
    def _is_complete_adapter(path: Path) -> bool:
        """Return ``True`` if ``path`` contains a complete LoRA adapter."""
        has_config = (path / "adapter_config.json").is_file()
        has_weights = (
            (path / "adapter_model.safetensors").is_file()
            or (path / "adapter_model.bin").is_file()
        )
        return has_config and has_weights

    def _resolve_tokenizer_source(self, adapter_source: str | Path | None) -> str:
        """Choose the best tokenizer source."""
        base_model_name: str = self.config["model"]["base_model"]

        if adapter_source is None:
            return base_model_name

        # Local path: prefer adapter tokenizer if it ships one
        adapter_path = Path(str(adapter_source))
        if adapter_path.is_dir() and (adapter_path / "tokenizer_config.json").is_file():
            logger.info("Using tokenizer from local adapter: %s", adapter_source)
            return str(adapter_source)

        # HF Hub ID: try to load from the adapter repo (may have custom tokens)
        if isinstance(adapter_source, str):
            logger.info("Using tokenizer from HF Hub adapter repo: %s", adapter_source)
            return adapter_source

        return base_model_name

    def _load_tokenizer(self, tokenizer_source: str) -> None:
        """Load and configure the tokenizer."""
        base_model_name: str = self.config["model"]["base_model"]
        logger.info("Loading tokenizer from: %s", tokenizer_source)
        start = time.perf_counter()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        except Exception:
            logger.warning(
                "Tokenizer load failed from %s → falling back to: %s",
                tokenizer_source,
                base_model_name,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Tokenizer loaded in %.2f s.", time.perf_counter() - start)

    def _load_base_model(self) -> AutoModelForCausalLM:
        """Load the base causal language model with memory-aware settings."""
        base_model_name: str = self.config["model"]["base_model"]
        logger.info("Loading base model: %s", base_model_name)
        start = time.perf_counter()

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        inference_config = self.config.get("inference", {})

        max_memory: dict = {"cpu": inference_config.get("max_cpu_memory", "12GiB")}
        if torch.cuda.is_available():
            max_memory[0] = inference_config.get("max_gpu_memory", "4GiB")

        offload_dir = _PROJECT_ROOT / inference_config.get(
            "offload_directory", "data/model_offload"
        )
        offload_dir.mkdir(parents=True, exist_ok=True)

        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                device_map="auto",
                max_memory=max_memory,
                offload_folder=offload_dir,
                offload_state_dict=True,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load base model '{base_model_name}': {e}"
            ) from e

        logger.info("Base model loaded in %.2f s.", time.perf_counter() - start)
        return base_model

    def _apply_lora(
        self,
        base_model: AutoModelForCausalLM,
        adapter_source: str | Path | None,
    ) -> None:
        """Apply the LoRA adapter to the base model, or use base model as-is."""
        if not self.use_lora or adapter_source is None:
            self.model = base_model
            logger.info("LoRA disabled — using base model only.")
            return

        logger.info("Applying LoRA adapter from: %s", adapter_source)
        start = time.perf_counter()

        try:
            self.model = PeftModel.from_pretrained(
                base_model, str(adapter_source), is_trainable=False
            )
            logger.info("LoRA adapter applied in %.2f s.", time.perf_counter() - start)
        except Exception as e:
            logger.warning(
                "Failed to load LoRA adapter from %s: %s — falling back to base model.",
                adapter_source,
                e,
            )
            self.model = base_model
