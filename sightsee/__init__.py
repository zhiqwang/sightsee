import argparse
import inspect
import os
import socket
import sys
from pathlib import Path

from netron.server import __version__, start, wait

AUTO_HOST = "auto"


def _is_usable_ipv4(host):
    return bool(host) and host not in {"0.0.0.0", "127.0.0.1"} and not host.startswith("127.")


def _detect_default_host():
    env_host = os.getenv("SIGHTSEE_HOST", "").strip()
    if env_host:
        return env_host

    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        host = sock.getsockname()[0]
        if _is_usable_ipv4(host):
            return host
    except OSError:
        pass
    finally:
        if sock is not None:
            sock.close()

    try:
        host = socket.gethostbyname(socket.gethostname())
        if _is_usable_ipv4(host):
            return host
    except OSError:
        pass

    return None


def _resolve_host(host):
    if host in (None, AUTO_HOST):
        return _detect_default_host()
    return host


def _build_address(host=None, port=None):
    if host is not None:
        return (host, port)
    return port


def _build_start_kwargs(start_fn, browse=False, verbosity=None):
    kwargs = {"browse": browse}
    if verbosity is None:
        return kwargs

    try:
        parameters = inspect.signature(start_fn).parameters
    except (TypeError, ValueError):
        parameters = {}

    if "verbosity" in parameters:
        kwargs["verbosity"] = verbosity

    return kwargs


def _format_access_url(address, fallback_host=None):
    host, port = address
    if host in {None, "", "0.0.0.0", "::"}:
        host = fallback_host or _detect_default_host() or "localhost"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def main():
    parser = argparse.ArgumentParser(description="Viewer for neural network and deep learning models.")
    parser.add_argument("file", metavar="MODEL_FILE", help="model file to serve", nargs="?", default=None)
    parser.add_argument("-b", "--browse", help="launch web browser", action="store_true")
    parser.add_argument("-p", "--port", help="port to serve", type=int)
    parser.add_argument(
        "--host",
        metavar="ADDR",
        help="host to serve; defaults to auto-detected server IP, falling back to localhost",
        default=AUTO_HOST,
    )
    parser.add_argument(
        "--verbosity",
        metavar="LEVEL",
        help="output verbosity (quiet, default, debug)",
        choices=["quiet", "default", "debug", "0", "1", "2"],
        default=None,
    )
    parser.add_argument("--version", help="print version", action="store_true")
    args = parser.parse_args()

    if args.version:
        print(__version__)
        sys.exit(0)

    if args.file and not Path(args.file).exists():
        print(f"Model file '{args.file}' does not exist.")
        sys.exit(2)

    address = _build_address(host=_resolve_host(args.host), port=args.port)
    start_kwargs = _build_start_kwargs(start, browse=args.browse, verbosity=args.verbosity)
    final_address = start(args.file, address=address, **start_kwargs)
    print(f"Access URL: {_format_access_url(final_address, fallback_host=_resolve_host(args.host))}", flush=True)
    wait()
    sys.exit(0)
