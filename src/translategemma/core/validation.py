"""Validation of translation request payloads."""

from __future__ import annotations

import base64
import io
import warnings
from typing import Any, Dict

from translategemma.core.config import Config
from translategemma.core.errors import ValidationError

_SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF"})


def _parse_languages(data: Dict[str, Any], config: Config) -> tuple[str, str]:
    source_lang = data.get("source_lang", "English")
    target_lang = data.get("target_lang", "Vietnamese")
    for field_name, value in (
        ("source_lang", source_lang),
        ("target_lang", target_lang),
    ):
        if not isinstance(value, str):
            raise ValidationError(f"Field '{field_name}' must be a string")
        value = value.strip()
        if not value or len(value) > 64:
            raise ValidationError(
                f"Field '{field_name}' must contain 1-64 characters"
            )
        if any(ch in value for ch in "\r\n\x00"):
            raise ValidationError(f"Field '{field_name}' contains invalid characters")
        if field_name == "source_lang":
            source_lang = value
        else:
            target_lang = value
    return source_lang, target_lang


def _parse_max_tokens(data: Dict[str, Any], config: Config) -> int:
    raw_tokens = data.get("max_new_tokens", config.default_output_tokens)
    if isinstance(raw_tokens, bool):
        raise ValidationError("Field 'max_new_tokens' must be an integer")
    try:
        max_tokens = int(raw_tokens)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Field 'max_new_tokens' must be an integer") from exc
    if not 1 <= max_tokens <= config.max_output_tokens:
        raise ValidationError(
            f"Field 'max_new_tokens' must be between 1 and "
            f"{config.max_output_tokens}"
        )
    return max_tokens


def parse_translation_payload(data: Any, config: Config) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValidationError("Request body must be a JSON object")

    text = data.get("text")
    if not isinstance(text, str):
        raise ValidationError("Field 'text' must be a string")
    text = text.strip()
    if not text:
        raise ValidationError("Field 'text' must not be empty")
    if len(text) > config.max_input_chars:
        raise ValidationError(
            f"Field 'text' exceeds {config.max_input_chars} characters"
        )

    source_lang, target_lang = _parse_languages(data, config)
    max_tokens = _parse_max_tokens(data, config)

    return {
        "text": text,
        "src": source_lang,
        "tgt": target_lang,
        "max_tokens": max_tokens,
    }


def _decode_and_validate_image(
    encoded_bytes: bytes, config: Config
) -> Any:
    """Decode an image after validating its header, without decompression bombs."""
    from PIL import Image, ImageOps
    import numpy as np

    try:
        # Opening only parses the header; pixel data is not decompressed yet.
        # Pillow can flag a decompression bomb while opening/identifying the
        # image, so the filter must cover Image.open() as well as load().
        # Pillow's own MAX_IMAGE_PIXELS protection stays enabled; its warning
        # is treated as a hard error here instead of disabling it globally.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)

            with Image.open(io.BytesIO(encoded_bytes)) as img:
                if img.format not in _SUPPORTED_IMAGE_FORMATS:
                    raise ValidationError(
                        f"Unsupported image format: {img.format or 'unknown'}"
                    )

                header_width, header_height = img.size
                if header_width < 8 or header_height < 8:
                    raise ValidationError("Image is too small (min 8x8 pixels)")
                if (
                    header_width > config.max_image_width
                    or header_height > config.max_image_height
                ):
                    raise ValidationError(
                        f"Image dimensions exceed the maximum "
                        f"{config.max_image_width}x{config.max_image_height} pixels"
                    )
                if header_width * header_height > config.max_image_pixels:
                    raise ValidationError(
                        f"Image is too large (max {config.max_image_pixels} pixels)"
                    )

                img.load()

                # Apply EXIF orientation so the RGB array matches the intended view.
                oriented = ImageOps.exif_transpose(img)
                if oriented is not img:
                    width, height = oriented.size
                    if width < 8 or height < 8:
                        raise ValidationError("Image is too small (min 8x8 pixels)")
                    if (
                        width > config.max_image_width
                        or height > config.max_image_height
                    ):
                        raise ValidationError(
                            f"Image dimensions exceed the maximum "
                            f"{config.max_image_width}x{config.max_image_height} pixels"
                        )
                rgb = oriented.convert("RGB")
                # The inference engine ultimately resizes to 896x896. Bounding the
                # queued NumPy array here avoids retaining tens/hundreds of MB per job.
                if max(rgb.size) > 2048:
                    rgb.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                return np.asarray(rgb, dtype=np.uint8)
    except ValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValidationError(
            "Image is too large (possible decompression bomb)"
        ) from exc
    except Exception as exc:
        raise ValidationError("Field 'image' is not a decodable image") from exc


def parse_image_translation_payload(data: Any, config: Config) -> Dict[str, Any]:
    if not config.vision_enabled:
        raise ValidationError("Image translation is not enabled on this server")

    if not isinstance(data, dict):
        raise ValidationError("Request body must be a JSON object")

    raw = data.get("image")
    if not isinstance(raw, str) or not raw.strip():
        raise ValidationError("Field 'image' must be a base64-encoded image string")
    raw = raw.strip()

    try:
        encoded_bytes = base64.b64decode(raw, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValidationError("Field 'image' is not valid base64") from exc
    if not encoded_bytes:
        raise ValidationError("Field 'image' must not be empty")
    if len(encoded_bytes) > config.max_image_bytes:
        raise ValidationError(
            f"Field 'image' exceeds {config.max_image_bytes} bytes after decoding"
        )

    image = _decode_and_validate_image(encoded_bytes, config)

    source_lang, target_lang = _parse_languages(data, config)
    max_tokens = _parse_max_tokens(data, config)

    return {
        "text": "",
        "src": source_lang,
        "tgt": target_lang,
        "max_tokens": max_tokens,
        "image": image,
    }
