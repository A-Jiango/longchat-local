# Put Your Local Model Here

This directory is the default model location used by the app:

```text
longchat-local/model/
```

When you run `python chat.py` from the repository root, the backend loads `./model` unless you set `LLM_MODEL_PATH` in `.env`.

Copy the contents of one compatible local JANG / MLX vision-language model into this folder. The model files should be directly inside `model/`, not nested one level deeper.

Expected shape:

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

If your model lives somewhere else, set an absolute path in `.env`:

```env
LLM_MODEL_PATH=/absolute/path/to/model
```

Do not commit model weights unless you intentionally distribute them.
