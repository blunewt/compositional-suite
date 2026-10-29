"""The Windows icon must use the classic layout so Explorer renders every size."""
import importlib.util
import os
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_ico_uses_bitmap_frames_below_256_and_png_at_256():
    data = open(os.path.join(ROOT, "packaging", "icon.ico"), "rb").read()
    n = struct.unpack("<HHH", data[:6])[2]
    sizes = []
    for i in range(n):
        w, _h, _cc, _r, _planes, bpp, _size, offset = struct.unpack("<BBBBHHII", data[6 + 16 * i: 22 + 16 * i])
        px = w or 256
        sizes.append(px)
        assert bpp == 32
        if px < 256:
            assert struct.unpack("<I", data[offset:offset + 4])[0] == 40         # plain bitmap frame
        else:
            assert data[offset:offset + 4] == b"\x89PNG"
    assert sizes == [16, 24, 32, 48, 64, 128, 256]


def test_icon_check_script_only_reports(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("check_exe_icon", os.path.join(ROOT, "packaging", "check_exe_icon.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(str(tmp_path / "does_not_exist.exe")) == 0          # a diagnostic never fails a release
