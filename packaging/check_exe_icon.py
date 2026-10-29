"""
Reports whether a Windows executable carries icon resources (diagnostic only).

    python packaging/check_exe_icon.py dist/CompositionalSuite.exe

Compares the number of embedded icon images with packaging/icon.ico: a different
count means PyInstaller kept its default icon. This script only reports; it never
fails the build. Needs pefile (installed together with PyInstaller).
"""
import os
import struct
import sys

RT_ICON, RT_GROUP_ICON = 3, 14


def count_resources(exe):
    import pefile
    pe = pefile.PE(exe, fast_load=True)
    pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]])
    counts = {RT_ICON: 0, RT_GROUP_ICON: 0}
    root = getattr(pe, "DIRECTORY_ENTRY_RESOURCE", None)
    if root is not None:
        for entry in root.entries:
            if entry.id in counts and hasattr(entry, "directory"):
                counts[entry.id] = len(entry.directory.entries)
    return counts[RT_ICON], counts[RT_GROUP_ICON]


def main(exe):
    try:
        icons, groups = count_resources(exe)
        ico = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
        expected = struct.unpack("<HHH", open(ico, "rb").read(6))[2]
    except Exception as exc:                      # a diagnostic must never break a release
        print(f"Could not inspect {exe}: {exc}")
        return 0
    print(f"{os.path.basename(exe)}: {icons} icon image(s) in {groups} icon group(s); packaging/icon.ico has {expected}")
    if icons == 0:
        print("WARNING: the executable has no icon at all.")
    elif icons != expected:
        print("WARNING: the icon count differs from packaging/icon.ico - the default PyInstaller icon may still be in use.")
    else:
        print("OK: the logo icon is embedded.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
