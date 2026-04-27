import argparse
import unittest.mock

import pytest

import sightsee


# --- _is_usable_ipv4 ---


@pytest.mark.parametrize(
    "host, expected",
    [
        (None, False),
        ("", False),
        ("0.0.0.0", False),
        ("127.0.0.1", False),
        ("127.0.1.1", False),
        ("127.255.255.255", False),
        ("192.168.1.1", True),
        ("10.0.0.1", True),
        ("172.16.0.1", True),
    ],
)
def test_is_usable_ipv4(host, expected):
    assert sightsee._is_usable_ipv4(host) is expected


# --- _build_onnx_identifier ---


def test_build_onnx_identifier_replaces_existing_suffix():
    assert sightsee._build_onnx_identifier("/tmp/model.plan") == "model.onnx"


def test_build_onnx_identifier_appends_suffix_when_missing():
    assert sightsee._build_onnx_identifier("/tmp/model") == "model.onnx"


def test_build_onnx_identifier_with_hidden_file():
    # Python treats ".plan" as a name with no suffix, so stem=".plan"
    assert sightsee._build_onnx_identifier("/tmp/.plan") == ".plan.onnx"


def test_build_onnx_identifier_with_nested_path():
    assert sightsee._build_onnx_identifier("/a/b/c/deep_model.trt") == "deep_model.onnx"


# --- _should_use_onnx_identifier ---


def test_should_use_onnx_identifier_for_as_onnx():
    assert sightsee._should_use_onnx_identifier("/tmp/model.plan", as_onnx=True) is True


def test_should_use_onnx_identifier_for_axmodel_suffix():
    assert sightsee._should_use_onnx_identifier("/tmp/model.axmodel") is True
    assert sightsee._should_use_onnx_identifier("/tmp/model.AXMODEL") is True


def test_should_not_use_onnx_identifier_for_regular_onnx_suffix():
    assert sightsee._should_use_onnx_identifier("/tmp/model.onnx") is False


# --- _AliasedContentProvider ---


def test_aliased_content_provider_exposes_identifier_as_data_path(tmp_path):
    model_path = tmp_path / "model.axmodel"
    model_path.write_bytes(b"onnx-data")

    content = sightsee._AliasedContentProvider(str(model_path), "model.onnx", str(model_path))

    assert content.base == "model.onnx"
    assert content.identifier == "model.onnx"
    assert content.name == str(model_path)
    assert content.read("model.onnx") == b"onnx-data"


# --- _detect_default_host ---


def test_detect_default_host_uses_env_override(monkeypatch):
    monkeypatch.setenv("SIGHTSEE_HOST", "10.0.0.9")
    assert sightsee._detect_default_host() == "10.0.0.9"


def test_detect_default_host_ignores_whitespace_only_env(monkeypatch):
    monkeypatch.setenv("SIGHTSEE_HOST", "   ")

    class FakeSocket:
        def connect(self, address):
            pass

        def getsockname(self):
            return ("10.5.5.5", 12345)

        def close(self):
            pass

    monkeypatch.setattr(sightsee.socket, "socket", lambda *a, **kw: FakeSocket())
    assert sightsee._detect_default_host() == "10.5.5.5"


def test_detect_default_host_uses_udp_probe(monkeypatch):
    class FakeSocket:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def connect(self, address):
            assert address == ("8.8.8.8", 80)

        def getsockname(self):
            return ("192.168.10.22", 54321)

        def close(self):
            self.closed = True

    monkeypatch.delenv("SIGHTSEE_HOST", raising=False)
    monkeypatch.setattr(sightsee.socket, "socket", lambda *args, **kwargs: FakeSocket())
    assert sightsee._detect_default_host() == "192.168.10.22"


def test_list_interface_ipv4_hosts_uses_psutil(monkeypatch):
    fake_psutil = unittest.mock.Mock()
    fake_psutil.net_if_addrs.return_value = {
        "lo": [unittest.mock.Mock(family=sightsee.socket.AF_INET, address="127.0.0.1")],
        "Ethernet": [
            unittest.mock.Mock(family=sightsee.socket.AF_INET, address="10.20.30.40"),
            unittest.mock.Mock(family=sightsee.socket.AF_INET6, address="fe80::1"),
        ],
        "docker0": [unittest.mock.Mock(family=sightsee.socket.AF_INET, address="172.17.0.1")],
        "Wi-Fi": [unittest.mock.Mock(family=sightsee.socket.AF_INET, address=" 192.168.1.9 ")],
    }
    fake_psutil.net_if_stats.return_value = {
        "lo": unittest.mock.Mock(isup=True),
        "Ethernet": unittest.mock.Mock(isup=True),
        "docker0": unittest.mock.Mock(isup=False),
        "Wi-Fi": unittest.mock.Mock(isup=True),
    }
    monkeypatch.setattr(sightsee, "psutil", fake_psutil)

    assert sightsee._list_interface_ipv4_hosts() == [
        ("Ethernet", "10.20.30.40"),
        ("Wi-Fi", "192.168.1.9"),
    ]


def test_pick_interface_ipv4_prefers_physical_interface():
    hosts = [
        ("docker0", "172.17.0.1"),
        ("tailscale0", "100.64.0.5"),
        ("enp68s0", "10.20.30.40"),
    ]
    assert sightsee._pick_interface_ipv4(hosts) == "10.20.30.40"


def test_detect_default_host_uses_interface_when_hostname_is_loopback(monkeypatch):
    class BrokenSocket:
        def connect(self, address):
            raise PermissionError("denied")

        def close(self):
            pass

    monkeypatch.delenv("SIGHTSEE_HOST", raising=False)
    monkeypatch.setattr(sightsee.socket, "socket", lambda *args, **kwargs: BrokenSocket())
    monkeypatch.setattr(sightsee, "_list_interface_ipv4_hosts", lambda: [("docker0", "172.17.0.1"), ("enp68s0", "10.5.5.5")])
    monkeypatch.setattr(sightsee.socket, "gethostname", lambda: "server01")
    monkeypatch.setattr(sightsee.socket, "gethostbyname", lambda hostname: "127.0.1.1")
    assert sightsee._detect_default_host() == "10.5.5.5"


def test_detect_default_host_falls_back_to_hostname(monkeypatch):
    class BrokenSocket:
        def connect(self, address):
            raise PermissionError("denied")

        def close(self):
            pass

    monkeypatch.delenv("SIGHTSEE_HOST", raising=False)
    monkeypatch.setattr(sightsee.socket, "socket", lambda *args, **kwargs: BrokenSocket())
    monkeypatch.setattr(sightsee, "_list_interface_ipv4_hosts", lambda: [])
    monkeypatch.setattr(sightsee.socket, "gethostname", lambda: "server01")
    monkeypatch.setattr(sightsee.socket, "gethostbyname", lambda hostname: "10.12.0.7")
    assert sightsee._detect_default_host() == "10.12.0.7"


def test_detect_default_host_returns_none_on_loopback(monkeypatch):
    class BrokenSocket:
        def connect(self, address):
            raise PermissionError("denied")

        def close(self):
            pass

    monkeypatch.delenv("SIGHTSEE_HOST", raising=False)
    monkeypatch.setattr(sightsee.socket, "socket", lambda *args, **kwargs: BrokenSocket())
    monkeypatch.setattr(sightsee, "_list_interface_ipv4_hosts", lambda: [])
    monkeypatch.setattr(sightsee.socket, "gethostname", lambda: "server01")
    monkeypatch.setattr(sightsee.socket, "gethostbyname", lambda hostname: "127.0.1.1")
    assert sightsee._detect_default_host() is None


def test_detect_default_host_returns_none_when_all_fail(monkeypatch):
    """Returns None when both the UDP probe and hostname lookup raise OSError."""

    class BrokenSocket:
        def connect(self, address):
            raise OSError("no network")

        def close(self):
            pass

    monkeypatch.delenv("SIGHTSEE_HOST", raising=False)
    monkeypatch.setattr(sightsee.socket, "socket", lambda *a, **kw: BrokenSocket())
    monkeypatch.setattr(sightsee, "_list_interface_ipv4_hosts", lambda: [])
    monkeypatch.setattr(sightsee.socket, "gethostname", lambda: "host")
    def _raise_oserror(h):
        raise OSError("no dns")

    monkeypatch.setattr(sightsee.socket, "gethostbyname", _raise_oserror)
    assert sightsee._detect_default_host() is None


def test_build_address_defaults_to_none():
    assert sightsee._build_address() is None


def test_build_address_with_host_only():
    assert sightsee._build_address(host="0.0.0.0") == ("0.0.0.0", None)


def test_build_address_with_port_only():
    assert sightsee._build_address(port=8080) == 8080


def test_build_start_kwargs_skips_unsupported_verbosity():
    def current_start(file=None, address=None, browse=True):
        return (file, address, browse)

    assert sightsee._build_start_kwargs(current_start, browse=True, verbosity="debug") == {"browse": True}


def test_build_start_kwargs_passes_supported_verbosity():
    def legacy_start(file=None, address=None, browse=True, verbosity="default"):
        return (file, address, browse, verbosity)

    assert sightsee._build_start_kwargs(legacy_start, browse=False, verbosity="quiet") == {
        "browse": False,
        "verbosity": "quiet",
    }


def test_resolve_host_supports_auto(monkeypatch):
    monkeypatch.setattr(sightsee, "_detect_default_host", lambda: "10.1.2.3")
    assert sightsee._resolve_host(sightsee.AUTO_HOST) == "10.1.2.3"


def test_format_access_url_prefers_bound_host():
    assert sightsee._format_access_url(("10.20.30.40", 8080)) == "http://10.20.30.40:8080"


def test_format_access_url_uses_fallback_for_wildcard_bind():
    assert sightsee._format_access_url(("0.0.0.0", 8080), fallback_host="10.20.30.40") == "http://10.20.30.40:8080"


def test_main_uses_generated_model_path_with_auto_host(tmp_path, write_test_model, monkeypatch, capsys):
    model_path = write_test_model(tmp_path / "test_model.onnx")
    calls = {}

    def fake_start_model(file=None, address=None, browse=True, verbosity=None, identifier=None):
        calls["file"] = file
        calls["address"] = address
        calls["browse"] = browse
        calls["verbosity"] = verbosity
        calls["identifier"] = identifier
        return ("10.10.10.20", 8080)

    monkeypatch.setattr(
        sightsee.argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            file=model_path,
            browse=False,
            port=None,
            host=sightsee.AUTO_HOST,
            verbosity=None,
            as_onnx=False,
            version=False,
        ),
    )
    monkeypatch.setattr(sightsee, "_detect_default_host", lambda: "10.10.10.20")
    monkeypatch.setattr(sightsee, "_start_model", fake_start_model)
    monkeypatch.setattr(sightsee, "wait", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        sightsee.main()

    assert exc_info.value.code == 0
    assert calls == {
        "file": model_path,
        "address": ("10.10.10.20", None),
        "browse": False,
        "verbosity": None,
        "identifier": None,
    }
    assert capsys.readouterr().out.strip() == "Access URL: http://10.10.10.20:8080"


def test_main_rejects_missing_model(tmp_path, monkeypatch, capsys):
    missing_path = str(tmp_path / "missing.onnx")

    monkeypatch.setattr(
        sightsee.argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            file=missing_path,
            browse=False,
            port=None,
            host=sightsee.AUTO_HOST,
            verbosity=None,
            as_onnx=False,
            version=False,
        ),
    )
    monkeypatch.setattr(sightsee.Path, "exists", lambda self: False)

    with pytest.raises(SystemExit) as exc_info:
        sightsee.main()

    assert exc_info.value.code == 2
    assert capsys.readouterr().out.strip() == f"Model file '{missing_path}' does not exist."


def test_main_rejects_as_onnx_without_model(monkeypatch, capsys):
    monkeypatch.setattr(
        sightsee.argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            file=None,
            browse=False,
            port=None,
            host=sightsee.AUTO_HOST,
            verbosity=None,
            as_onnx=True,
            version=False,
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        sightsee.main()

    assert exc_info.value.code == 2
    assert capsys.readouterr().out.strip() == "Option '--as-onnx' requires MODEL_FILE."


def test_main_passes_onnx_identifier_override(tmp_path, write_test_model, monkeypatch, capsys):
    model_path = write_test_model(tmp_path / "test_model.model")
    calls = {}

    def fake_start_model(file=None, address=None, browse=True, verbosity=None, identifier=None):
        calls["file"] = file
        calls["address"] = address
        calls["browse"] = browse
        calls["verbosity"] = verbosity
        calls["identifier"] = identifier
        return ("10.10.10.20", 8080)

    monkeypatch.setattr(
        sightsee.argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            file=model_path,
            browse=True,
            port=9000,
            host="0.0.0.0",
            verbosity="debug",
            as_onnx=True,
            version=False,
        ),
    )
    monkeypatch.setattr(sightsee, "_start_model", fake_start_model)
    monkeypatch.setattr(sightsee, "wait", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        sightsee.main()

    assert exc_info.value.code == 0
    assert calls == {
        "file": model_path,
        "address": ("0.0.0.0", 9000),
        "browse": True,
        "verbosity": "debug",
        "identifier": "test_model.onnx",
    }
    assert capsys.readouterr().out.strip() == "Access URL: http://10.10.10.20:8080"


# --- _build_address (additional) ---


def test_build_address_with_host_and_port():
    assert sightsee._build_address(host="10.0.0.1", port=9090) == ("10.0.0.1", 9090)


# --- _build_start_kwargs (additional) ---


def test_build_start_kwargs_returns_early_when_verbosity_is_none():
    """When verbosity is None, return only browse without inspecting the function."""

    def any_start(file=None, address=None, browse=True, verbosity="default"):
        pass

    assert sightsee._build_start_kwargs(any_start, browse=True, verbosity=None) == {"browse": True}


def test_build_start_kwargs_handles_uninspectable_function(monkeypatch):
    """When signature inspection fails, verbosity is silently dropped."""

    def dummy_start():
        pass

    def _raise_typeerror(fn):
        raise TypeError("no sig")

    monkeypatch.setattr(sightsee.inspect, "signature", _raise_typeerror)
    result = sightsee._build_start_kwargs(dummy_start, browse=False, verbosity="debug")
    assert result == {"browse": False}


# --- _resolve_host (additional) ---


def test_resolve_host_with_none(monkeypatch):
    monkeypatch.setattr(sightsee, "_detect_default_host", lambda: "10.20.30.40")
    assert sightsee._resolve_host(None) == "10.20.30.40"


def test_resolve_host_with_explicit_host():
    assert sightsee._resolve_host("192.168.1.100") == "192.168.1.100"


# --- _format_access_url (additional) ---


def test_format_access_url_wraps_ipv6_in_brackets():
    """IPv6 addresses containing ':' should be wrapped in brackets."""
    assert sightsee._format_access_url(("fe80::1", 8080)) == "http://[fe80::1]:8080"


def test_format_access_url_does_not_double_wrap_ipv6():
    """Already bracketed IPv6 should not be double-wrapped."""
    assert sightsee._format_access_url(("[::1]", 8080)) == "http://[::1]:8080"


def test_format_access_url_empty_host_uses_fallback(monkeypatch):
    monkeypatch.setattr(sightsee, "_detect_default_host", lambda: None)
    assert sightsee._format_access_url(("", 3000)) == "http://localhost:3000"


def test_format_access_url_none_host_no_fallback(monkeypatch):
    monkeypatch.setattr(sightsee, "_detect_default_host", lambda: None)
    assert sightsee._format_access_url((None, 5000)) == "http://localhost:5000"


def test_format_access_url_double_colon_host_uses_fallback():
    assert sightsee._format_access_url(("::", 8080), fallback_host="10.0.0.1") == "http://10.0.0.1:8080"


# --- _start_model ---


def test_start_model_without_identifier_delegates_to_start(monkeypatch):
    """When identifier is None, _start_model delegates to netron's start."""
    captured = {}

    def fake_start(file, address=None, **kwargs):
        captured["file"] = file
        captured["address"] = address
        captured["kwargs"] = kwargs
        return ("0.0.0.0", 9999)

    monkeypatch.setattr(sightsee, "start", fake_start)
    result = sightsee._start_model(file="model.onnx", address=8080, browse=False, verbosity=None, identifier=None)
    assert result == ("0.0.0.0", 9999)
    assert captured["file"] == "model.onnx"
    assert captured["address"] == 8080


def test_start_model_with_identifier_starts_custom_server(tmp_path, write_test_model, monkeypatch):
    """When identifier is provided, _start_model uses the custom ContentProvider path."""
    model_path = write_test_model(tmp_path / "model.plan")

    fake_content = unittest.mock.MagicMock()
    monkeypatch.setattr(sightsee.netron_server, "_ContentProvider", lambda *args: fake_content)
    monkeypatch.setattr(sightsee.netron_server, "_make_address", lambda addr: ("0.0.0.0", 0))
    monkeypatch.setattr(sightsee.netron_server, "_make_port", lambda addr: ("0.0.0.0", 7777))

    fake_thread = unittest.mock.MagicMock()
    alive_calls = [False, True]  # first call returns False, second returns True
    fake_thread.alive.side_effect = lambda: alive_calls.pop(0) if alive_calls else True
    fake_thread.url = "http://0.0.0.0:7777"
    monkeypatch.setattr(sightsee.netron_server, "_HTTPServerThread", lambda content, addr: fake_thread)
    monkeypatch.setattr(sightsee.webbrowser, "open", lambda url: None)
    monkeypatch.setattr(sightsee.time, "sleep", lambda _: None)

    # Ensure root logger has no handlers to cover basicConfig branch (line 89)
    root_logger = sightsee.logging.getLogger()
    original_handlers = root_logger.handlers[:]
    root_logger.handlers.clear()
    try:
        result = sightsee._start_model(
            file=model_path, address=("0.0.0.0", 0), browse=False, verbosity=None, identifier="model.onnx"
        )
    finally:
        root_logger.handlers[:] = original_handlers

    assert result == ("0.0.0.0", 7777)
    fake_thread.start.assert_called_once()


def test_start_model_with_identifier_and_browse(tmp_path, write_test_model, monkeypatch):
    """When browse=True and identifier is set, webbrowser.open is called."""
    model_path = write_test_model(tmp_path / "model.plan")

    monkeypatch.setattr(sightsee.netron_server, "_ContentProvider", lambda *args: unittest.mock.MagicMock())
    monkeypatch.setattr(sightsee.netron_server, "_make_address", lambda addr: ("0.0.0.0", 0))
    monkeypatch.setattr(sightsee.netron_server, "_make_port", lambda addr: ("0.0.0.0", 7778))

    fake_thread = unittest.mock.MagicMock()
    fake_thread.alive.return_value = True
    fake_thread.url = "http://0.0.0.0:7778"
    monkeypatch.setattr(sightsee.netron_server, "_HTTPServerThread", lambda content, addr: fake_thread)

    opened_urls = []
    monkeypatch.setattr(sightsee.webbrowser, "open", lambda url: opened_urls.append(url))

    sightsee._start_model(
        file=model_path, address=("0.0.0.0", 0), browse=True, verbosity=None, identifier="model.onnx"
    )
    assert opened_urls == ["http://0.0.0.0:7778"]


def test_start_model_with_identifier_nonzero_port_calls_stop(tmp_path, write_test_model, monkeypatch):
    """When address has a non-zero port, netron_server.stop is called."""
    model_path = write_test_model(tmp_path / "model.plan")

    monkeypatch.setattr(sightsee.netron_server, "_ContentProvider", lambda *args: unittest.mock.MagicMock())
    monkeypatch.setattr(sightsee.netron_server, "_make_address", lambda addr: ("0.0.0.0", 9000))

    stopped = []
    monkeypatch.setattr(sightsee.netron_server, "stop", lambda addr: stopped.append(addr))

    fake_thread = unittest.mock.MagicMock()
    fake_thread.alive.return_value = True
    fake_thread.url = "http://0.0.0.0:9000"
    monkeypatch.setattr(sightsee.netron_server, "_HTTPServerThread", lambda content, addr: fake_thread)
    monkeypatch.setattr(sightsee.webbrowser, "open", lambda url: None)

    sightsee._start_model(
        file=model_path, address=("0.0.0.0", 9000), browse=False, verbosity=None, identifier="model.onnx"
    )
    assert stopped == [("0.0.0.0", 9000)]


def test_start_model_with_identifier_raises_on_missing_file(tmp_path):
    """When identifier is set but file doesn't exist, FileNotFoundError is raised."""
    missing = str(tmp_path / "nonexistent.plan")
    with pytest.raises(FileNotFoundError):
        sightsee._start_model(file=missing, identifier="model.onnx")


# --- main (additional) ---


def test_main_prints_version(monkeypatch, capsys):
    """--version flag prints version and exits with code 0."""
    monkeypatch.setattr(
        sightsee.argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            file=None,
            browse=False,
            port=None,
            host=sightsee.AUTO_HOST,
            verbosity=None,
            as_onnx=False,
            version=True,
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        sightsee.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out.strip()
    assert output == sightsee.__version__


# --- __main__.py ---


def test_main_module_entry_point():
    """Running `python -m sightsee` calls main()."""
    import runpy

    with unittest.mock.patch("sightsee.main") as mock_main:
        runpy.run_module("sightsee", run_name="__main__")
    mock_main.assert_called_once()
