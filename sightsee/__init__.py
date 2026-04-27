import argparse
import inspect
import logging
import os
import socket
import sys
import time
import webbrowser
from pathlib import Path

import psutil

from netron import server as netron_server
from netron.server import __version__, start, wait

AUTO_HOST = "auto"
ONNX_COMPATIBLE_SUFFIXES = {".axmodel"}
PREFERRED_INTERFACE_PREFIXES = ("eth", "en", "bond", "wlan", "wl", "wwan", "ethernet", "wi-fi", "wifi")
DEPRIORITIZED_INTERFACE_MARKERS = (
    "br-",
    "docker",
    "virbr",
    "veth",
    "cni",
    "flannel",
    "cali",
    "tailscale",
    "zerotier",
    "zt",
    "wg",
    "tun",
    "tap",
    "vboxnet",
    "vmnet",
    "vethernet",
    "hyper-v",
    "default switch",
    "virtual",
)


def _is_usable_ipv4(host):
    return bool(host) and host not in {"0.0.0.0", "127.0.0.1"} and not host.startswith("127.")


def _interface_is_up(stats):
    return stats is None or getattr(stats, "isup", True)


def _list_interface_ipv4_hosts():
    try:
        interface_addrs = psutil.net_if_addrs()
    except OSError:
        return []

    try:
        interface_stats = psutil.net_if_stats()
    except OSError:
        interface_stats = {}

    hosts = []
    for name, addrs in interface_addrs.items():
        if not _interface_is_up(interface_stats.get(name)):
            continue
        for addr in addrs:
            if addr.family != socket.AF_INET:
                continue
            host = addr.address.strip() if isinstance(addr.address, str) else addr.address
            if _is_usable_ipv4(host):
                hosts.append((name, host))

    return hosts


def _interface_priority(name):
    lowered = name.lower()
    if lowered == "lo":
        return 100
    if lowered.startswith(PREFERRED_INTERFACE_PREFIXES):
        return 0
    if any(marker in lowered for marker in DEPRIORITIZED_INTERFACE_MARKERS):
        return 80
    return 40


def _pick_interface_ipv4(hosts):
    if not hosts:
        return None
    return min(hosts, key=lambda item: (_interface_priority(item[0]), item[0], item[1]))[1]


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

    interface_host = _pick_interface_ipv4(_list_interface_ipv4_hosts())
    if interface_host:
        return interface_host

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


def _should_use_onnx_identifier(file, as_onnx=False):
    return bool(as_onnx or (file and Path(file).suffix.lower() in ONNX_COMPATIBLE_SUFFIXES))


class _AliasedContentProvider(netron_server._ContentProvider):
    def __init__(self, path, identifier, name):
        super().__init__(None, path, identifier, name)
        self._real_base = self.base
        self.base = os.path.basename(identifier)

    def read(self, path):
        if path == self.base:
            path = self._real_base
        return super().read(path)


def _start_model(file=None, address=None, browse=True, verbosity=None, identifier=None):
    if identifier is None:
        start_kwargs = _build_start_kwargs(start, browse=browse, verbosity=verbosity)
        return start(file, address=address, **start_kwargs)

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if file and not os.path.exists(file):
        raise FileNotFoundError(file)

    # Keep the real path for file access, but expose an ONNX-looking data URL for Netron format matching.
    content = _AliasedContentProvider(file, identifier, file)

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
    identifier = _build_onnx_identifier(args.file) if _should_use_onnx_identifier(args.file, args.as_onnx) else None
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
