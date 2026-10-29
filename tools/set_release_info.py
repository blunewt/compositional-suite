"""
Fills the release placeholders in one step.

    python tools/set_release_info.py --org <github-user-or-org>

Replaces TODO-org in the repository URLs (README, pyproject, CITATION.cff,
package metadata). The LICENSE already names its copyright holder; only if it
still contains the "TODO: COPYRIGHT HOLDER" placeholder, add
--holder "<copyright holder>" to fill it. Run
`python tools/check_placeholders.py` afterwards; it must report nothing before
a release tag is pushed.
"""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_FILES = ["README.md", "pyproject.toml", "CITATION.cff", "compositional_suite/__init__.py"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--org", required=True, help="GitHub user or organisation that owns the repository")
    ap.add_argument("--holder", help='copyright holder, only needed if LICENSE still has the placeholder')
    a = ap.parse_args()
    for rel in URL_FILES:
        p = os.path.join(ROOT, rel)
        s = open(p, encoding="utf-8").read()
        n = s.count("TODO-org")
        open(p, "w", encoding="utf-8").write(s.replace("TODO-org", a.org))
        print(f"{rel}: {n} URL(s) updated")
    p = os.path.join(ROOT, "LICENSE")
    s = open(p, encoding="utf-8").read()
    if "TODO: COPYRIGHT HOLDER" in s:
        if not a.holder:
            raise SystemExit("LICENSE still has the copyright-holder placeholder: pass --holder")
        open(p, "w", encoding="utf-8").write(s.replace("TODO: COPYRIGHT HOLDER", a.holder))
        print("LICENSE: copyright holder set")
    else:
        print("LICENSE: already complete")


if __name__ == "__main__":
    main()
