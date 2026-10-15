# Exercise #8: Image Generator CLI

`generate.py` generates images from a text prompt with OpenAI's `gpt-image-2`
and saves every image locally under a safe, unique filename.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # then put your real OPENAI_API_KEY in .env
```

## Usage

```powershell
python generate.py `
    --prompt "A red wooden cottage beside a Finnish lake" `
    --aspect-ratio 16:9 `
    --count 3 `
    --output images
```

| Flag | Default | Description |
|---|---|---|
| `--prompt` | *(required)* | What to generate (must not be empty) |
| `--aspect-ratio` | `1:1` | `W:H` (e.g. `1:1`, `3:2`, `2:3`, `16:9`) or `auto`. The model only supports `1024x1024`, `1536x1024` and `1024x1536`, so the closest size is used and a note is printed |
| `--count` | `1` | Number of images, 1–10 |
| `--output` | `images` | Output directory (created if missing) |
| `--quality` | model default | `auto`, `low`, `medium`, `high` |
| `--negative-prompt` | – | Only for models that support it (see below) |
| `--seed` | – | Only for models that support it (see below) |
| `--model` | `gpt-image-2` | Also `gpt-image-1`, `gpt-image-1-mini` |
| `--format` | `png` | `png`, `jpeg`/`jpg`, `webp` |

### Model capabilities

Each model has an entry in the `MODEL_CAPABILITIES` table in `generate.py`.
OpenAI's GPT Image models do **not** offer a negative prompt or a seed, so
`--negative-prompt` and `--seed` stop with a clear error instead of being
ignored without warning. A model or provider that supports them only needs its
table entry updated.

## What the program does

1. **Validates** the arguments with argparse and checks them against the
   selected model's capabilities. Invalid input exits with code 2.
2. **Calls** `client.images.generate(...)` with a timeout and automatic retries.
3. **Handles errors** such as a bad API key, rate limits or no remaining quota,
   timeouts, network failures, prompts rejected by safety checks, other HTTP
   errors and file-write errors. Each one prints a short message and exits
   with code 1.
4. **Generates the requested count**. If the API returns fewer images than
   requested, it asks for the missing ones.
5. **Saves every image locally**. `extract_image_bytes()` accepts Base64 data
   (what OpenAI returns), temporary URLs (downloaded right away), file-like
   objects or raw bytes, so it does not rely on a permanent URL.
6. **Uses safe, unique filenames** in the form
   `YYYYMMDD-HHMMSS_<prompt-slug>_<index>_<random8>.<ext>`. The prompt slug
   contains only ASCII `a-z0-9-`, so characters like `/`, `..` or `ä` cannot
   create bad paths. Files are opened in exclusive-create mode so an existing
   file is never overwritten.
7. **Prints** the full path of each saved file and a summary.
