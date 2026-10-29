"""The windowed executable has no console: start-up errors must reach a log file.
Tests never open a real dialog (a modal message box would hang a CI run)."""
import ctypes
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_launcher():
    spec = importlib.util.spec_from_file_location("cs_launcher", os.path.join(ROOT, "packaging", "launcher.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _isolate(monkeypatch, tmp_path):
    for var in ("LOCALAPPDATA", "HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(tmp_path))


def test_start_up_errors_are_written_to_a_log(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    _load_launcher()._report_fatal("Traceback (most recent call last):\nValueError: boom", show_dialog=False)
    log = tmp_path / "CompositionalSuite" / "crash.log"
    assert log.exists() and "ValueError: boom" in log.read_text(encoding="utf-8")


def test_windows_shows_a_message_box_with_the_error(tmp_path, monkeypatch):
    """Simulated Windows: a fake MessageBoxW stands in for the real, blocking one."""
    _isolate(monkeypatch, tmp_path)
    shown = []

    class FakeUser32:
        @staticmethod
        def MessageBoxW(hwnd, text, title, flags):
            shown.append(text)
            return 1

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", type("W", (), {"user32": FakeUser32})(), raising=False)
    _load_launcher()._report_fatal("ValueError: boom")
    assert shown and "ValueError: boom" in shown[0] and "crash.log" in shown[0]


def test_no_dialog_when_disabled_even_on_windows(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    called = []

    class FakeUser32:
        @staticmethod
        def MessageBoxW(*args):
            called.append(args)

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", type("W", (), {"user32": FakeUser32})(), raising=False)
    _load_launcher()._report_fatal("x", show_dialog=False)
    assert not called
