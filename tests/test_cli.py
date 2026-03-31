import argparse

import pytest

import sightsee


def test_detect_default_host_uses_env_override(monkeypatch):
    monkeypatch.setenv("SIGHTSEE_HOST", "10.0.0.9")
    assert sightsee._detect_default_host() == "10.0.0.9"


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


def test_detect_default_host_falls_back_to_hostname(monkeypatch):
    class BrokenSocket:
        def connect(self, address):
            raise PermissionError("denied")

        def close(self):
            pass

    monkeypatch.delenv("SIGHTSEE_HOST", raising=False)
    monkeypatch.setattr(sightsee.socket, "socket", lambda *args, **kwargs: BrokenSocket())
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
    monkeypatch.setattr(sightsee.socket, "gethostname", lambda: "server01")
    monkeypatch.setattr(sightsee.socket, "gethostbyname", lambda hostname: "127.0.1.1")
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

    def fake_start(file=None, address=None, browse=True):
        calls["file"] = file
        calls["address"] = address
        calls["browse"] = browse
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
            version=False,
        ),
    )
    monkeypatch.setattr(sightsee, "_detect_default_host", lambda: "10.10.10.20")
    monkeypatch.setattr(sightsee, "start", fake_start)
    monkeypatch.setattr(sightsee, "wait", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        sightsee.main()

    assert exc_info.value.code == 0
    assert calls == {
        "file": model_path,
        "address": ("10.10.10.20", None),
        "browse": False,
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
            version=False,
        ),
    )
    monkeypatch.setattr(sightsee.Path, "exists", lambda self: False)

    with pytest.raises(SystemExit) as exc_info:
        sightsee.main()

    assert exc_info.value.code == 2
    assert capsys.readouterr().out.strip() == f"Model file '{missing_path}' does not exist."
