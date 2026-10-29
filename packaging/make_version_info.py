"""
Writes the Windows version resource for PyInstaller (--version-file) from the
package metadata, so the executable's Properties > Details show the product
name, version and DTU copyright.

    python packaging/make_version_info.py build/version_info.txt

Reads compositional_suite/__init__.py as text (no import needed).
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def metadata():
    text = open(os.path.join(ROOT, "compositional_suite", "__init__.py"), encoding="utf-8").read()
    version = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M).group(1)
    copyright_ = re.search(r'^__copyright__\s*=\s*"([^"]+)"', text, re.M).group(1)
    copyright_ = copyright_.encode("ascii").decode("unicode_escape")      # \u00a9 -> ©
    return version, copyright_


def version_info_text(version, copyright_):
    nums = [int(x) for x in re.findall(r"\d+", version)[:4]]
    nums += [0] * (4 - len(nums))
    vt = tuple(nums)
    # ASCII-only file: non-ASCII characters are written as escapes
    lit = lambda s: repr(s).encode("ascii", "backslashreplace").decode("ascii")
    return f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={vt},
    prodvers={vt},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '040904B0',
        [StringStruct('CompanyName', 'Technical University of Denmark'),
         StringStruct('FileDescription', 'Compositional Suite'),
         StringStruct('FileVersion', {lit(version)}),
         StringStruct('InternalName', 'CompositionalSuite'),
         StringStruct('LegalCopyright', {lit(copyright_)}),
         StringStruct('OriginalFilename', 'CompositionalSuite.exe'),
         StringStruct('ProductName', 'Compositional Suite'),
         StringStruct('ProductVersion', {lit(version)})])
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "version_info.txt")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="ascii") as fh:
        fh.write(version_info_text(*metadata()))
    print(f"wrote {out}")
