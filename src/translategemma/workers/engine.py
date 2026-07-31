"""TranslateGemma inference engine that lives inside one isolated GPU process."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from translategemma.workers.generation import GenerationPlan, plan_generation

logger = logging.getLogger("translategemma")


def _resolve_keras_classes() -> tuple[Any, Any, Any]:
    import keras_hub

    try:
        models = keras_hub.models
        return (
            models.Gemma3Backbone,
            models.Gemma3CausalLMPreprocessor,
            models.Gemma3CausalLM,
        )
    except AttributeError:
        # Compatibility fallback for older Kaggle images.
        from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
        from keras_hub.src.models.gemma3.gemma3_causal_lm import Gemma3CausalLM
        from keras_hub.src.models.gemma3.gemma3_causal_lm_preprocessor import (
            Gemma3CausalLMPreprocessor,
        )

        return Gemma3Backbone, Gemma3CausalLMPreprocessor, Gemma3CausalLM


class TranslateGemmaEngine:
    """Lives only inside one isolated GPU process."""

    VISION_TOKEN_COUNT = 256
    GENERATION_SAFETY_TOKENS = 8

    def __init__(
        self,
        preset_path: str,
        dtype: str,
        vision_enabled: bool = False,
        *,
        generation_bucketing: bool = True,
        generation_length_buckets: Iterable[int] = (256, 512, 1024, 1536, 2048),
        generation_bucket_step: int = 512,
        warmup_output_tokens: int = 128,
        warmup_text_buckets: Iterable[int] = (256,),
        warmup_vision_buckets: Iterable[int] = (512,),
        compilation_cache_dir: Optional[str] = None,
    ):
        self.preset_path = preset_path
        self.dtype = dtype
        self.vision_enabled = vision_enabled
        self.generation_bucketing = bool(generation_bucketing)
        self.generation_length_buckets = tuple(sorted(set(generation_length_buckets)))
        self.generation_bucket_step = int(generation_bucket_step)
        self.warmup_output_tokens = int(warmup_output_tokens)
        self.warmup_text_buckets = tuple(sorted(set(warmup_text_buckets)))
        self.warmup_vision_buckets = tuple(sorted(set(warmup_vision_buckets)))
        self.compilation_cache_dir = compilation_cache_dir
        self.model: Any = None
        self.preprocessor: Any = None

    def load(self, warmup_enabled: bool) -> Dict[str, Any]:
        import jax
        import keras
        import keras_hub

        Gemma3Backbone, Gemma3CausalLMPreprocessor, Gemma3CausalLM = (
            _resolve_keras_classes()
        )

        preset = Path(self.preset_path)
        required_files = ["config.json", "preprocessor.json", "model.weights.h5"]
        for filename in required_files:
            if not (preset / filename).is_file():
                raise FileNotFoundError(f"Missing {filename} in {self.preset_path}")
        spm = preset / "assets" / "tokenizer" / "vocabulary.spm"
        if not spm.is_file():
            raise FileNotFoundError(f"Missing tokenizer file: {spm}")

        started = time.time()
        mode = "multimodal (vision + text)" if self.vision_enabled else "text-only"
        logger.info("Building %s Gemma3 backbone", mode)

        backbone_doc = json.loads((preset / "config.json").read_text(encoding="utf-8"))
        backbone_cfg = dict(backbone_doc["config"])
        if not self.vision_enabled:
            backbone_cfg["vision_encoder"] = None
        backbone = Gemma3Backbone.from_config(backbone_cfg)
        backbone.load_weights(str(preset / "model.weights.h5"), skip_mismatch=True)
        logger.info("Backbone and weights loaded in %.1fs", time.time() - started)

        preprocessor_doc = json.loads(
            (preset / "preprocessor.json").read_text(encoding="utf-8")
        )
        preprocessor_cfg = dict(preprocessor_doc["config"])
        if self.vision_enabled:
            # Pad (rather than center-crop) non-square images so that text near
            # the edges is preserved instead of being cropped away.
            image_converter_cfg = dict(preprocessor_cfg["image_converter"]["config"])
            image_converter_cfg["crop_to_aspect_ratio"] = False
            image_converter_cfg["pad_to_aspect_ratio"] = True
            preprocessor_cfg["image_converter"] = {
                "module": "keras_hub.src.models.gemma3.gemma3_image_converter",
                "class_name": "Gemma3ImageConverter",
                "config": image_converter_cfg,
            }
        else:
            preprocessor_cfg["image_converter"] = None
        preprocessor = Gemma3CausalLMPreprocessor.from_config(preprocessor_cfg)
        preprocessor.tokenizer.set_proto(str(spm))

        self.model = Gemma3CausalLM(preprocessor, backbone, dtype=self.dtype)
        self.preprocessor = self.model.preprocessor

        warmup_metadata: Dict[str, Any] = {}
        if warmup_enabled:
            warmup_metadata = self._warmup()
            logger.info("Warm-up buckets completed: %s", warmup_metadata)

        devices = [str(device) for device in jax.devices()]
        metadata = {
            "load_seconds": round(time.time() - started, 3),
            "devices": devices,
            "keras_version": getattr(keras, "__version__", "unknown"),
            "keras_hub_version": getattr(keras_hub, "__version__", "unknown"),
            "jax_version": getattr(jax, "__version__", "unknown"),
            "dtype": self.dtype,
            "vision_enabled": self.vision_enabled,
            "generation_bucketing": self.generation_bucketing,
            "generation_length_buckets": list(self.generation_length_buckets),
            "generation_bucket_step": self.generation_bucket_step,
            "compilation_cache_dir": self.compilation_cache_dir,
            "warmup_enabled": warmup_enabled,
            "warmup_output_tokens": self.warmup_output_tokens,
            "warmup": warmup_metadata,
        }
        logger.info("Model ready: %s", metadata)
        return metadata

    def _warmup(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"text": [], "vision": []}
        if not self.generation_bucketing:
            logger.info("Generation bucketing disabled; warming exact default shapes")
            started = time.time()
            self.translate(
                "Hello, world!",
                "English",
                "Vietnamese",
                self.warmup_output_tokens,
            )
            metadata["text"].append(
                {"max_length": "exact", "seconds": round(time.time() - started, 3)}
            )
            if self.vision_enabled:
                started = time.time()
                self.translate_image(
                    self._blank_image(),
                    "English",
                    "Vietnamese",
                    self.warmup_output_tokens,
                )
                metadata["vision"].append(
                    {"max_length": "exact", "seconds": round(time.time() - started, 3)}
                )
            return metadata

        text_prompt = self._text_prompt("Hello, world!", "English", "Vietnamese")
        text_prompt_tokens = self._token_count(text_prompt)
        for bucket in self.warmup_text_buckets:
            exact = (
                text_prompt_tokens
                + self.warmup_output_tokens
                + self.GENERATION_SAFETY_TOKENS
            )
            if bucket < exact:
                logger.warning(
                    "Skipping text warm-up bucket=%d; needs at least %d tokens",
                    bucket,
                    exact,
                )
                continue
            logger.info("Compiling text generation bucket max_length=%d", bucket)
            started = time.time()
            self._translate_text(
                "Hello, world!",
                "English",
                "Vietnamese",
                self.warmup_output_tokens,
                forced_max_length=bucket,
            )
            metadata["text"].append(
                {"max_length": bucket, "seconds": round(time.time() - started, 3)}
            )

        if self.vision_enabled:
            vision_prompt = self._vision_prompt("English", "Vietnamese")
            vision_prompt_tokens = self._token_count(vision_prompt)
            for bucket in self.warmup_vision_buckets:
                exact = (
                    vision_prompt_tokens
                    + self.VISION_TOKEN_COUNT
                    + self.warmup_output_tokens
                    + self.GENERATION_SAFETY_TOKENS
                )
                if bucket < exact:
                    logger.warning(
                        "Skipping vision warm-up bucket=%d; needs at least %d tokens",
                        bucket,
                        exact,
                    )
                    continue
                logger.info("Compiling vision generation bucket max_length=%d", bucket)
                started = time.time()
                self._translate_image(
                    self._blank_image(),
                    "English",
                    "Vietnamese",
                    self.warmup_output_tokens,
                    forced_max_length=bucket,
                )
                metadata["vision"].append(
                    {"max_length": bucket, "seconds": round(time.time() - started, 3)}
                )
        return metadata

    @staticmethod
    def _blank_image() -> Any:
        """Small blank image used for the vision warm-up request."""
        import numpy as np

        return np.full((64, 64, 3), 255, dtype=np.uint8)

    @staticmethod
    def _text_prompt(text: str, src: str, tgt: str) -> str:
        return f"user: Translate from {src} to {tgt}:\n{text.strip()}\nmodel: "

    @staticmethod
    def _vision_prompt(src: str, tgt: str) -> str:
        return (
            f"user: Translate the text in the image from {src} to {tgt}:\n"
            f"<start_of_image>\nmodel: "
        )

    def _token_count(self, text: str) -> int:
        return len(self.preprocessor.tokenizer(text))

    def _plan(
        self,
        *,
        prompt_tokens: int,
        max_tokens: int,
        vision_tokens: int = 0,
    ) -> GenerationPlan:
        return plan_generation(
            prompt_tokens=prompt_tokens,
            requested_new_tokens=max_tokens,
            vision_tokens=vision_tokens,
            safety_tokens=self.GENERATION_SAFETY_TOKENS,
            bucketing_enabled=self.generation_bucketing,
            buckets=self.generation_length_buckets,
            fallback_step=self.generation_bucket_step,
        )

    def translate(self, text: str, src: str, tgt: str, max_tokens: int) -> str:
        return self._translate_text(text, src, tgt, max_tokens)

    def _translate_text(
        self,
        text: str,
        src: str,
        tgt: str,
        max_tokens: int,
        *,
        forced_max_length: Optional[int] = None,
    ) -> str:
        prompt = self._text_prompt(text, src, tgt)
        prompt_length = self._token_count(prompt)
        plan = self._plan(prompt_tokens=prompt_length, max_tokens=max_tokens)
        compiled_max_length = forced_max_length or plan.compiled_max_length
        if compiled_max_length < plan.exact_max_length:
            raise ValueError(
                f"Compiled max_length {compiled_max_length} is below required "
                f"length {plan.exact_max_length}"
            )
        self._log_plan("text", plan, compiled_max_length, forced_max_length is not None)
        output = self.model.generate(
            prompt,
            max_length=compiled_max_length,
            stop_token_ids="auto",
            strip_prompt=True,
        )
        return self._finalize_output(str(output), max_tokens, compiled_max_length != plan.exact_max_length)

    def translate_image(
        self, image: Any, src: str, tgt: str, max_tokens: int
    ) -> str:
        return self._translate_image(image, src, tgt, max_tokens)

    def _translate_image(
        self,
        image: Any,
        src: str,
        tgt: str,
        max_tokens: int,
        *,
        forced_max_length: Optional[int] = None,
    ) -> str:
        prompt = self._vision_prompt(src, tgt)
        prompt_length = self._token_count(prompt)
        plan = self._plan(
            prompt_tokens=prompt_length,
            max_tokens=max_tokens,
            vision_tokens=self.VISION_TOKEN_COUNT if self.vision_enabled else 0,
        )
        compiled_max_length = forced_max_length or plan.compiled_max_length
        if compiled_max_length < plan.exact_max_length:
            raise ValueError(
                f"Compiled max_length {compiled_max_length} is below required "
                f"length {plan.exact_max_length}"
            )
        self._log_plan("vision", plan, compiled_max_length, forced_max_length is not None)
        output = self.model.generate(
            {"prompts": prompt, "images": image},
            max_length=compiled_max_length,
            stop_token_ids="auto",
            strip_prompt=True,
        )
        return self._finalize_output(str(output), max_tokens, compiled_max_length != plan.exact_max_length)

    @staticmethod
    def _log_plan(
        mode: str,
        plan: GenerationPlan,
        compiled_max_length: int,
        forced: bool,
    ) -> None:
        logger.info(
            "Generation plan mode=%s prompt_tokens=%d vision_tokens=%d "
            "requested_new=%d exact_max_length=%d compiled_max_length=%d "
            "bucketed=%s forced=%s",
            mode,
            plan.prompt_tokens,
            plan.vision_tokens,
            plan.requested_new_tokens,
            plan.exact_max_length,
            compiled_max_length,
            compiled_max_length != plan.exact_max_length,
            forced,
        )

    def _finalize_output(self, text: str, max_tokens: int, bucket_has_slack: bool) -> str:
        cleaned = self._clean(text)
        if not bucket_has_slack or not cleaned:
            return cleaned
        # KerasHub CausalLM exposes max *total* length rather than max-new-tokens.
        # Bucketing can therefore leave extra generation room.  Re-tokenize only
        # bucketed outputs and hard-trim to the client's requested new-token cap.
        tokens = self.preprocessor.tokenizer.tokenize(cleaned)
        if len(tokens) <= max_tokens:
            return cleaned
        trimmed = tokens[:max_tokens]
        decoded = self.preprocessor.tokenizer.detokenize(trimmed)
        result = self._scalar_text(decoded)
        logger.warning(
            "Trimmed bucketed completion to requested max_new_tokens=%d", max_tokens
        )
        return self._clean(result)

    @staticmethod
    def _scalar_text(value: Any) -> str:
        if hasattr(value, "numpy"):
            value = value.numpy()
        if hasattr(value, "item") and not isinstance(value, (bytes, str)):
            try:
                value = value.item()
            except (ValueError, TypeError):
                pass
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @staticmethod
    def _clean(text: str) -> str:
        result = text.strip()
        for marker in ("<end_of_turn>", "<eos>", "</s>"):
            result = result.replace(marker, "")
        return result.strip()
