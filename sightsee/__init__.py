import argparse
import socket
import sys
from pathlib import Path

from netron.server import __version__, start, wait

NETRON_CONFIG_PATH = Path.home() / ".netronrc"


def set_config():
    # host name
    host_name = socket.gethostname()
    # ip address
    host_ip = socket.gethostbyname(host_name)
    with open(NETRON_CONFIG_PATH, "w") as f:
        f.write(host_ip)
        f.write("\n")


def get_config():
    if not NETRON_CONFIG_PATH.exists():
        set_config()

    with open(NETRON_CONFIG_PATH, "r") as f:
        host_ip = f.read().strip()

    return host_ip


def main():
    parser = argparse.ArgumentParser(description="Viewer for neural network and deep learning models.")
    parser.add_argument("file", metavar="MODEL_FILE", help="model file to serve", nargs="?", default=None)
    parser.add_argument("-b", "--browse", help="launch web browser", action="store_true")
    parser.add_argument("-p", "--port", help="port to serve", type=int)
    parser.add_argument("--host", metavar="ADDR", help="host to serve", default=get_config())
    parser.add_argument(
        "--verbosity",
        metavar="LEVEL",
        help="output verbosity (quiet, default, debug)",
        choices=["quiet", "default", "debug", "0", "1", "2"],
        default="default",
    )
    parser.add_argument("--version", help="print version", action="store_true")
    args = parser.parse_args()

    if args.file and not Path(args.file).exists():
        print(f"Model file '{args.file}' does not exist.")
        sys.exit(2)

    if args.version:
        print(__version__)
        sys.exit(0)

    address = (args.host, args.port) if args.host else args.port if args.port else get_config()
    start(args.file, address=address, browse=args.browse, verbosity=args.verbosity)
    wait()
    sys.exit(0)
