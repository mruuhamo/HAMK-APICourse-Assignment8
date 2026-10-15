"""Exercise #8: Image Generator CLI.

Generates one or more images from a text prompt with an OpenAI image model
(default: gpt-image-2), saves every image locally under a safe, unique
filename and prints the saved paths.

Example:
    python generate.py --prompt "A red wooden cottage beside a Finnish lake" \
        --aspect-ratio 16:9 --count 3 --output images
"""

from __future__ import annotations

import argparse
import base64
import binascii
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_MODEL = "gpt-image-2"
DEFAULT_OUTPUT_DIR = "images"
MAX_COUNT = 10
MAX_PROMPT_LENGTH = 32_000
DOWNLOAD_TIMEOUT_S = 60
REQUEST_TIMEOUT_S = 300


@dataclass(frozen=True)
class ModelCapabilities:
    sizes: dict[str, tuple[int, int]]
    qualities: tuple[str, ...]
    formats: tuple[str, ...]
    max_n: int
    supports_negative_prompt: bool = False
    supports_seed: bool = False


GPT_IMAGE_SIZES = {
    "1024x1024": (1024, 1024),
    "1536x1024": (1536, 1024),
    "1024x1536": (1024, 1536),
}

MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
    "gpt-image-2": ModelCapabilities(
        sizes=GPT_IMAGE_SIZES,
        qualities=("auto", "low", "medium", "high"),
        formats=("png", "jpeg", "webp"),
        max_n=10,
    ),
    "gpt-image-1": ModelCapabilities(
        sizes=GPT_IMAGE_SIZES,
        qualities=("auto", "low", "medium", "high"),
        formats=("png", "jpeg", "webp"),
        max_n=10,
    ),
    "gpt-image-1-mini": ModelCapabilities(
        sizes=GPT_IMAGE_SIZES,
        qualities=("auto", "low", "medium", "high"),
        formats=("png", "jpeg", "webp"),
        max_n=10,
    ),
}

FORMAT_EXTENSIONS = {"png": "png", "jpeg": "jpg", "webp": "webp"}


class GenerationError(RuntimeError):
    """Raised for user-facing generation failures."""


# ---------------------------------------------------------------- validation


def count_type(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer, got {value!r}")
    if not 1 <= number <= MAX_COUNT:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_COUNT}, got {number}")
    return number


def aspect_ratio_type(value: str) -> str | tuple[int, int]:
    text = value.strip().lower()
    if text == "auto":
        return "auto"
    match = re.fullmatch(r"(\d{1,4})\s*[:x/]\s*(\d{1,4})", text)
    if not match:
        raise argparse.ArgumentTypeError(
            f"expected W:H (e.g. 16:9, 1:1, 2:3) or 'auto', got {value!r}"
        )
    width, height = int(match.group(1)), int(match.group(2))
    if width == 0 or height == 0:
        raise argparse.ArgumentTypeError("aspect ratio parts must be greater than zero")
    if max(width, height) / min(width, height) > 4:
        raise argparse.ArgumentTypeError("aspect ratio is too extreme (max 4:1)")
    return width, height


def prompt_type(value: str) -> str:
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("must not be empty")
    if len(text) > MAX_PROMPT_LENGTH:
        raise argparse.ArgumentTypeError(f"must be at most {MAX_PROMPT_LENGTH} characters")
    return text


def seed_type(value: str) -> int:
    try:
        seed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer, got {value!r}")
    if not 0 <= seed <= 2**32 - 1:
        raise argparse.ArgumentTypeError("must be between 0 and 4294967295")
    return seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate images from a text prompt and save them locally.",
        epilog='Example: python generate.py --prompt "A red wooden cottage beside a '
        'Finnish lake" --aspect-ratio 16:9 --count 3 --output images',
    )
    parser.add_argument("--prompt", required=True, type=prompt_type, help="what to generate")
    parser.add_argument(
        "--aspect-ratio",
        type=aspect_ratio_type,
        default=(1, 1),
        help="W:H such as 1:1, 3:2, 2:3, 16:9 or 'auto' (default: 1:1); "
        "mapped to the closest size the model supports",
    )
    parser.add_argument(
        "--count", type=count_type, default=1, help=f"number of images, 1-{MAX_COUNT} (default: 1)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=sorted(MODEL_CAPABILITIES),
        help=f"image model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--quality", help="quality level, if supported by the model (e.g. low/medium/high/auto)")
    parser.add_argument("--negative-prompt", type=prompt_type, help="things to avoid, if supported by the model")
    parser.add_argument("--seed", type=seed_type, help="random seed, if supported by the model")
    parser.add_argument("--format", default="png", help="output image format (default: png)")
    return parser


def validate_against_model(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    caps = MODEL_CAPABILITIES[args.model]
    if args.negative_prompt is not None and not caps.supports_negative_prompt:
        parser.error(f"--negative-prompt is not supported by model {args.model}")
    if args.seed is not None and not caps.supports_seed:
        parser.error(f"--seed is not supported by model {args.model}")
    if args.quality is not None:
        args.quality = args.quality.strip().lower()
        if args.quality not in caps.qualities:
            parser.error(f"--quality for {args.model} must be one of: {', '.join(caps.qualities)}")
    args.format = args.format.strip().lower().replace("jpg", "jpeg")
    if args.format not in caps.formats:
        parser.error(f"--format for {args.model} must be one of: {', '.join(caps.formats)}")
    if args.output.exists() and not args.output.is_dir():
        parser.error(f"--output {args.output} exists and is not a directory")


def resolve_size(aspect: str | tuple[int, int], caps: ModelCapabilities) -> tuple[str, bool]:
    """Return (size string, exact_match) for the requested aspect ratio."""
    if aspect == "auto":
        return "auto", True
    target = aspect[0] / aspect[1]
    size, (w, h) = min(caps.sizes.items(), key=lambda item: abs(item[1][0] / item[1][1] - target))
    return size, abs(w / h - target) < 1e-6


# ---------------------------------------------------------- image extraction


def download_url(url: str) -> bytes:
    """Download a (possibly temporary) image URL immediately."""
    if not url.lower().startswith(("https://", "http://")):
        raise GenerationError(f"Refusing to download unsupported URL: {url[:60]}")
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise GenerationError(f"Could not download generated image: {exc}") from exc


def extract_image_bytes(item: Any) -> bytes:
    """Normalize a provider result item (Base64, URL, file-like or bytes) to bytes."""
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    if hasattr(item, "read"):
        return item.read()

    b64 = getattr(item, "b64_json", None) or (item.get("b64_json") if isinstance(item, dict) else None)
    if b64:
        try:
            return base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise GenerationError("Invalid Base64 image data received.") from exc

    url = getattr(item, "url", None) or (item.get("url") if isinstance(item, dict) else None)
    if url:
        return download_url(url)

    raise GenerationError("The API response contained no image data.")


# ------------------------------------------------------------------- saving


def slugify(text: str, max_length: int = 40) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "image"


def save_image(data: bytes, directory: Path, prompt: str, index: int, extension: str) -> Path:
    """Write bytes to a new, unique file; never overwrites existing files."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = slugify(prompt)
    for _ in range(5):
        path = directory / f"{stamp}_{slug}_{index:02d}_{uuid.uuid4().hex[:8]}.{extension}"
        try:
            with path.open("xb") as handle:
                handle.write(data)
            return path
        except FileExistsError:
            continue
    raise GenerationError("Could not create a unique filename.")


# --------------------------------------------------------------- generation


def generate_images(client: OpenAI, args: argparse.Namespace, size: str) -> list[bytes]:
    caps = MODEL_CAPABILITIES[args.model]
    images: list[bytes] = []
    attempts = 0
    while len(images) < args.count:
        attempts += 1
        if attempts > 3:
            raise GenerationError(
                f"API returned only {len(images)} of {args.count} requested images."
            )
        n = min(args.count - len(images), caps.max_n)
        request: dict[str, Any] = {
            "model": args.model,
            "prompt": args.prompt,
            "size": size,
            "n": n,
            "output_format": args.format,
        }
        if args.quality:
            request["quality"] = args.quality
        result = client.images.generate(**request)
        for item in result.data or []:
            images.append(extract_image_bytes(item))
    return images[: args.count]


def describe_api_error(exc: openai.OpenAIError) -> str:
    if isinstance(exc, openai.AuthenticationError):
        return "Authentication failed. Check OPENAI_API_KEY."
    if isinstance(exc, openai.PermissionDeniedError):
        return f"Permission denied (is your organization verified for this model?): {exc.message}"
    if isinstance(exc, openai.RateLimitError):
        return f"Rate limit or quota exceeded: {exc.message}"
    if isinstance(exc, openai.APITimeoutError):
        return "The request timed out. Try again or lower --count/--quality."
    if isinstance(exc, openai.APIConnectionError):
        return "Network error: could not reach the OpenAI API."
    if isinstance(exc, openai.BadRequestError):
        if "moderation" in str(exc).lower() or "safety" in str(exc).lower():
            return f"The prompt was rejected by the safety system: {exc.message}"
        return f"Invalid request: {exc.message}"
    if isinstance(exc, openai.APIStatusError):
        return f"API error (HTTP {exc.status_code}): {exc.message}"
    return f"OpenAI error: {exc}"


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_against_model(args, parser)

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set (environment or .env file).", file=sys.stderr)
        return 1

    caps = MODEL_CAPABILITIES[args.model]
    size, exact = resolve_size(args.aspect_ratio, caps)
    if not exact:
        w, h = args.aspect_ratio
        print(f"Note: {args.model} does not support {w}:{h}; using closest size {size}.")

    try:
        args.output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Error: cannot create output directory {args.output}: {exc}", file=sys.stderr)
        return 1

    print(f"Generating {args.count} image(s) with {args.model} ({size})...")
    client = OpenAI(timeout=REQUEST_TIMEOUT_S, max_retries=2)

    try:
        images = generate_images(client, args, size)
    except openai.OpenAIError as exc:
        print(f"Error: {describe_api_error(exc)}", file=sys.stderr)
        return 1
    except GenerationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    extension = FORMAT_EXTENSIONS[args.format]
    saved: list[Path] = []
    for index, data in enumerate(images, start=1):
        try:
            path = save_image(data, args.output, args.prompt, index, extension)
        except (OSError, GenerationError) as exc:
            print(f"Error: could not save image {index}: {exc}", file=sys.stderr)
            continue
        saved.append(path)
        print(f"Saved: {path.resolve()}")

    print(f"Done: {len(saved)}/{args.count} image(s) saved to {args.output.resolve()}")
    return 0 if len(saved) == args.count else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
