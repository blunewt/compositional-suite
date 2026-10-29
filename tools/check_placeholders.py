"""
Lists release placeholders that still need real values.

    python tools/check_placeholders.py          # report, exit 1 if any remain
    python tools/check_placeholders.py --warn   # report only, always exit 0

CI runs the strict form on tagged builds, so a release cannot be published
with "TODO-org" URLs or an unnamed copyright holder.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERNS = [r"TODO-org", r"TODO: COPYRIGHT HOLDER", r"TODO: add the citation", r"your-org"]
FILES = ["LICENSE", "pyproject.toml", "README.md", "CITATION.cff",
         "compositional_suite/__init__.py", "compositional_suite/app.py"]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    hits = []
    for rel in FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if any(re.search(p, line) for p in PATTERNS):
                    hits.append(f"{rel}:{n}: {line.strip()}")
    if hits:
        print("Release placeholders still present:")
        print("\n".join("  " + h for h in hits))
    else:
        print("No release placeholders found.")
    return 0 if (not hits or "--warn" in argv) else 1


if __name__ == "__main__":
    sys.exit(main())
