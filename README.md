# Sightsee

Sightsee is a thin CLI wrapper around `netron` for opening model files in a browser.

The repository is managed through `pyproject.toml` and `uv`.

## Development

```bash
uv sync --group dev
uv run pytest -q
```

## Release to PyPI

This repository publishes from GitHub Actions using PyPI API tokens.

One-time setup:

1. Create a project-scoped API token in Test PyPI and save it in this repository as `TEST_PYPI_API_TOKEN`.
2. Create a project-scoped API token in PyPI and save it in this repository as `PYPI_API_TOKEN`.
3. Keep the Git tag aligned with `[project].version` in `pyproject.toml`.

Release flow:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Pushing a `v*` tag, or publishing a GitHub release from a tag, runs tests, builds the wheel and sdist, checks that the
tag matches the package version, then uploads `dist/` to Test PyPI and PyPI.

## Usage

By default, Sightsee auto-detects the current server IPv4 address for binding by probing the active network interfaces.
If detection fails, it falls back to Netron's localhost behavior. You can override this with `--host` or the
`SIGHTSEE_HOST` environment variable.
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
uv run sightsee path/to/model.axmodel
uv run sightsee --as-onnx path/to/model.plan
```

Sightsee treats `.axmodel` files as ONNX-compatible by default. For other ONNX-compatible files without a `.onnx`
suffix, `--as-onnx` keeps the original file path for loading, but forces Netron to treat the input as an ONNX model.
