import argparse
import inspect
import logging
import os
import socket
import sys
import time
import webbrowser
from pathlib import Path

from netron import server as netron_server
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


def _build_onnx_identifier(file):
    name = Path(file).name
    stem = Path(name).stem if Path(name).suffix else name
    return f"{stem or 'model'}.onnx"


def _start_model(file=None, address=None, browse=True, verbosity=None, identifier=None):
    if identifier is None:
        start_kwargs = _build_start_kwargs(start, browse=browse, verbosity=verbosity)
        return start(file, address=address, **start_kwargs)

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if file and not os.path.exists(file):
        raise FileNotFoundError(file)

    # Keep the real path for file access, but override the identifier Netron uses for format matching.
    content = netron_server._ContentProvider(None, file, identifier, file)

    address = netron_server._make_address(address)
    if isinstance(address[1], int) and address[1] != 0:
        netron_server.stop(address)
    else:
        address = netron_server._make_port(address)

    thread = netron_server._HTTPServerThread(content, address)
    thread.start()
    while not thread.alive():
        time.sleep(0.01)

    if browse:
        webbrowser.open(thread.url)

    return address


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
    parser.add_argument(
        "--as-onnx",
        help="treat the input model as ONNX even when the file suffix is not .onnx",
        action="store_true",
    )
    parser.add_argument("--version", help="print version", action="store_true")
    args = parser.parse_args()

    if args.version:
        print(__version__)
        sys.exit(0)

    if args.as_onnx and not args.file:
        print("Option '--as-onnx' requires MODEL_FILE.")
        sys.exit(2)

    if args.file and not Path(args.file).exists():
        print(f"Model file '{args.file}' does not exist.")
        sys.exit(2)

    resolved_host = _resolve_host(args.host)
    address = _build_address(host=resolved_host, port=args.port)
    identifier = _build_onnx_identifier(args.file) if args.as_onnx and args.file else None
    final_address = _start_model(
        args.file,
        address=address,
        browse=args.browse,
        verbosity=args.verbosity,
        identifier=identifier,
    )
    print(f"Access URL: {_format_access_url(final_address, fallback_host=resolved_host)}", flush=True)
    wait()
    sys.exit(0)
