"""DTU owns software created by its employees: every place the software is made
available must state DTU as copyright holder."""
import importlib.util
import os
import sys
import types

import pytest

import compositional_suite as cs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTICE = "Copyright \u00a9 2026 Technical University of Denmark"


def test_notice_in_package_license_and_readme():
    assert cs.__copyright__ == NOTICE
    assert "Department of Health Technology" in cs.__developed_by__
    lic = open(os.path.join(ROOT, "LICENSE"), encoding="utf-8").read()
    assert NOTICE in lic and lic.count("Department of Health Technology") == 2
    assert NOTICE in open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_windows_version_resource_carries_the_notice(tmp_path):
    mvi = _load("make_version_info", ("packaging", "make_version_info.py"))
    version, notice = mvi.metadata()
    assert version == cs.__version__ and notice == NOTICE
    path = tmp_path / "version_info.txt"
    path.write_text(mvi.version_info_text(version, notice), encoding="ascii")
    pytest.importorskip("PyInstaller")
    import PyInstaller.compat as compat                     # its parser only needs win32api to write files
    if not hasattr(compat, "win32api"):
        compat.win32api = types.SimpleNamespace()
    from PyInstaller.utils.win32 import versioninfo
    vs = versioninfo.load_version_info_from_text_file(str(path))
    assert len(vs.toRaw()) > 0 and NOTICE in str(vs)
