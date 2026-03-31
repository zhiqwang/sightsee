# Sightsee

Sightsee is a thin CLI wrapper around `netron` for opening model files in a browser.

The repository is managed through `pyproject.toml` and `uv`.

## Development

```bash
uv sync --group dev
uv run pytest -q
```

## Usage

By default, Sightsee auto-detects the current server IPv4 address for binding. If detection fails, it falls back to
Netron's localhost behavior. You can override this with `--host` or the `SIGHTSEE_HOST` environment variable.
At startup, Sightsee prints a clear `Access URL:` line with the final address to open.

```bash
uv run sightsee path/to/model.onnx
```

Useful options:

```bash
uv run sightsee --browse path/to/model.onnx
uv run sightsee --host auto path/to/model.onnx
uv run sightsee --host 0.0.0.0 --port 8080 path/to/model.onnx
SIGHTSEE_HOST=10.10.0.5 uv run sightsee path/to/model.onnx
```
