# longchat-local

Local desktop chat for Apple Silicon, using a JANG / MLX vision-language model backend with structured long-context memory.

## What Is Included

- `chat.py` - application entry point.
- `chat_gui.py` - PySide6 desktop chat UI.
- `chat_backend.py` - local model session, streaming, prompt construction, context budgeting, and compression scheduling.
- `compression_cache.py` - structured compression, DAG indexing, and backtracking retrieval.
- `system_prompt.md` - default assistant system prompt.
- `assets/` - UI assets.
- `locales/` - language resources for compression heuristics.
- `scripts/peak_vram_probe.py` - machine-specific peak memory / prefill pressure probe.
- `.env.example` - deployment configuration template.
- `requirements.txt` - Python dependency list.

## Model Files

Model weights are excluded by design. A placeholder `model/README.md` is included only to show the expected location.

The default layout is:

```text
longchat-local/
  chat.py
  .env
  model/
    config.json
    tokenizer_config.json
    jang_config.json
    model.safetensors.index.json
    model-00001-of-00002.safetensors
    model-00002-of-00002.safetensors
    ...
```

Put the model files directly inside `model/`. Do not place them under `model/some-model-name/` unless `LLM_MODEL_PATH` points to that nested directory.

Alternatively, keep the model anywhere on disk and set an absolute path in `.env`:

```env
LLM_MODEL_PATH=/absolute/path/to/model
```

The current backend is wired for JANG / MLX VLM loading:

```python
from jang_tools.loader import load_jang_vlm_model
from mlx_vlm import stream_generate
```

This repository is directly reusable for compatible JANG / MLX vision-language models. Other model formats need a backend adapter in `ChatSession._ensure_runtime()`.

## Requirements

- Python 3.10 or newer.
- macOS on Apple Silicon for the included JANG / MLX backend.
- A local compatible JANG model directory.
- Enough unified memory for the model and chosen context budget.
- Optional for LaTeX math rendering in chat output:

```bash
brew install tectonic poppler
```

## Quick Start

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
cp .env.example .env
mkdir -p model
```

Place the compatible model files directly in `./model`, or edit `.env`:

```env
LLM_MODEL_PATH=/absolute/path/to/model
```

Quick model-path check:

```bash
test -f model/config.json && echo "model directory found"
```

Run the app:

```bash
python chat.py
```

## Optional English NLP Support

The app runs without spaCy or NLTK. For stronger English compression matching, install them after installing `requirements.txt`:

```bash
pip install "spacy>=3.7" "nltk>=3.8"
python -m spacy download en_core_web_sm
```

If spaCy, its English model, or NLTK is unavailable, the compression layer falls back to a lightweight tokenizer and built-in word variants.

## Recommended Memory Calibration

Before choosing a fixed context budget, run the probe:

```bash
python scripts/peak_vram_probe.py --mode auto --max-tokens 65536 --json-out peak_vram.json
```

Then set `LLM_CONTEXT_K_GB` conservatively in `.env`.

For example, on a 16GB machine, a reasonable first pass is:

```env
LLM_CONTEXT_K_GB=8
LLM_PREFILL_CONTEXT_CAP_TOKENS=null
```

If stable, try:

```env
LLM_CONTEXT_K_GB=10
LLM_PREFILL_CONTEXT_CAP_TOKENS=null
```

Using both values as `null` is safe but can be conservative because the framework estimates memory budget from observed runtime peaks rather than reading total available memory.

## Configuration Reference

The application reads `.env` from the repository root.

- `LLM_MODEL_PATH` - model directory. Relative paths are resolved from the repository root. The default `./model` means `longchat-local/model/`.
- `LLM_LOCALE` - language resource set for GUI text and compression heuristics. Bundled values: `zh-CN`, `en-US`.
- `LLM_CONTEXT_K_GB` - usable memory budget for context control. Use `null` for runtime auto-estimation.
- `LLM_PREFILL_CONTEXT_CAP_TOKENS` - fixed soft context token cap. Use `null` for speed/memory based auto mode. This is not a hard limit on the current user input; long input can elastically exceed it after history is compressed or trimmed.
- `LLM_STRUCTURED_COMPRESSION` - enable structured memory compression.
- `LLM_STRUCTURED_RECENT_TURNS` - recent raw turns kept outside structured compression.
- `LLM_STRUCTURED_MAX_DIRTY_BLOCKS` - number of dirty compression blocks updated per turn.
- `LLM_STRUCTURED_TARGET_TOKENS_PER_BLOCK` - target rendered size per compression block.
- `LLM_COMPRESSION_UPDATE_WAIT_SECONDS` - latency budget used to size compression updates.
- `LLM_TRACE_PREFILL` - print prefill and compression traces.

## Notes

- The repository is self-contained except for model weights and external runtime dependencies.
- The default model path is `./model` for portability.
- `model/README.md` is a placeholder; real model weights are ignored by `.gitignore` to avoid accidental commits.
- Natural-language compression rules live in `locales/*.json`; add another locale by copying one of those files and setting `LLM_LOCALE`.
- Keep model weights outside version control unless you intentionally distribute them.
