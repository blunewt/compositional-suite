"""
reactivity_library.py
---------------------
Read-only access to the reactivity-ratio library: the shipped file
(``compositional_suite/data/reactivity_ratios.csv``, the authors' curated table)
plus any local add-on files in the user library folder
(``~/.compositional_suite/libraries/*.csv``, or ``$COMPOSITIONAL_SUITE_LIBRARY_DIR``) in the
same CSV format. Add-on files stay on the user's computer.

Standard library only, so it is safe to import in batch workers and the CLI.

Typical use::

    from compositional_suite import reactivity_library as rl
    rl.find_monomers("methacryl")                 # -> ['2-Hydroxyethyl methacrylate', ...]
    rl.pair_entries("Styrene", "Methyl methacrylate")   # all usable determinations,
                                                        # oriented r_a -> Styrene
    R, sources, missing = rl.build_r_matrix(["Styrene", "Methyl methacrylate", "Acrylic acid"])

Orientation: every stored row is (monomer_1, r1, monomer_2, r2) exactly as in
its source. :func:`pair_entries` returns :class:`PairEntry` objects re-oriented
so that ``r_a`` always belongs to the *first* monomer you asked for.
"""

import csv
import functools
import os
import re
import shutil
from dataclasses import dataclass
from typing import Optional

try:  # Python >= 3.9
    from importlib.resources import files as _res_files
except ImportError:  # pragma: no cover
    _res_files = None

DATA_FILE = "reactivity_ratios.csv"

REQUIRED_COLUMNS = ("id", "monomer_1", "monomer_2", "r1", "r2", "usable", "dataset")

# Preference order when proposing a default entry for a pair.
DATASET_PRIORITY = {
    "polymer_handbook": 0,       # Polymer Handbook table, if added as a library file
    "own_curated": 1,            # the shipped curated library
    "textbook_table_6_1": 2,
    "legacy_reference": 3,
}
# Flags that demote an entry when choosing a default (it stays selectable).
SOFT_FLAGS = {
    "possible_autofill_doi", "same_values_as_other_reference",
    "atypical_value_check_source", "source_internal_inconsistency",
    "name_abbr_mismatch", "unsourced_rounded", "name_ocr_corrected",
}
REFERENCE_TEMPERATURE_C = 60.0


@dataclass(frozen=True)
class Entry:
    """One row of the library, as stored."""
    id: str
    monomer_1: str
    abbr_1: str
    monomer_2: str
    abbr_2: str
    r1: Optional[float]
    r2: Optional[float]
    r1_ci95: Optional[float]
    r2_ci95: Optional[float]
    temperature_C: str
    solvent: str
    conditions: str
    method: str
    model: str
    usable: bool
    flags: tuple
    reference: str
    doi: str
    dataset: str
    dataset_detail: str

    def temperature_value(self) -> Optional[float]:
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)", self.temperature_C or "")
        return float(m.group(1)) if m else None


@dataclass(frozen=True)
class PairEntry:
    """An :class:`Entry` oriented for a requested (monomer_a, monomer_b) pair."""
    entry: Entry
    monomer_a: str
    monomer_b: str
    r_a: float          # r_ab = k_aa / k_ab
    r_b: float          # r_ba = k_bb / k_ba

    def label(self) -> str:
        e = self.entry
        bits = [f"r_a={self.r_a:g}, r_b={self.r_b:g}"]
        if e.temperature_C:
            bits.append(f"{e.temperature_C} °C")
        if e.solvent:
            bits.append(e.solvent)
        if e.conditions:
            bits.append(e.conditions)
        bits.append(e.dataset.replace("_", " "))
        if e.flags:
            bits.append("⚠ " + ", ".join(e.flags))
        return " · ".join(bits)


def _f(value):
    try:
        return float(value) if str(value).strip() != "" else None
    except ValueError:
        return None


def default_path() -> str:
    if _res_files is not None:
        try:
            return str(_res_files("compositional_suite") / "data" / DATA_FILE)
        except (ModuleNotFoundError, TypeError):  # pragma: no cover
            pass
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", DATA_FILE)


def user_library_dir() -> str:
    """Folder whose ``*.csv`` files are loaded on top of the shipped library."""
    return (os.environ.get("COMPOSITIONAL_SUITE_LIBRARY_DIR")
            or os.path.join(os.path.expanduser("~"), ".compositional_suite", "libraries"))


def library_files() -> list:
    """Shipped file first, then local add-ons in name order."""
    files = [default_path()]
    folder = user_library_dir()
    if os.path.isdir(folder):
        files += sorted(os.path.join(folder, f) for f in os.listdir(folder)
                        if f.lower().endswith(".csv"))
    return files


def _read_file(path: str) -> list:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{os.path.basename(path)} is not a reactivity-ratio library "
                             f"file (missing columns: {', '.join(missing)}).")
        for rec in reader:
            g = lambda k: rec.get(k) or ""   # noqa: E731 - optional columns
            out.append(Entry(
                id=g("id"), monomer_1=g("monomer_1"), abbr_1=g("abbr_1"),
                monomer_2=g("monomer_2"), abbr_2=g("abbr_2"),
                r1=_f(g("r1")), r2=_f(g("r2")),
                r1_ci95=_f(g("r1_ci95")), r2_ci95=_f(g("r2_ci95")),
                temperature_C=g("temperature_C"), solvent=g("solvent"),
                conditions=g("conditions"), method=g("method"), model=g("model") or "terminal",
                usable=g("usable") == "yes",
                flags=tuple(x for x in g("flags").split(";") if x),
                reference=g("reference"), doi=g("doi"),
                dataset=g("dataset"), dataset_detail=g("dataset_detail"),
            ))
    return out


@functools.lru_cache(maxsize=4)
def load(path: Optional[str] = None) -> tuple:
    """
    Loads the library once and caches it. With no ``path``: the shipped file
    plus every local add-on (an add-on that fails to parse is skipped). Entry
    ids already seen are not loaded twice. Returns a tuple of :class:`Entry`.
    """
    paths = [path] if path else library_files()
    out, seen = [], set()
    for i, p in enumerate(paths):
        try:
            entries = _read_file(p)
        except (OSError, ValueError):
            if path or i == 0:
                raise
            continue
        for e in entries:
            if e.id not in seen:
                seen.add(e.id)
                out.append(e)
    return tuple(out)


def reload():
    """Clears the caches, e.g. after an add-on file was added."""
    load.cache_clear()
    _monomer_index.cache_clear()


def loaded_sources() -> list:
    """[(file name, number of entries)] for the shipped file and each add-on."""
    result = []
    for p in library_files():
        try:
            result.append((os.path.basename(p), len(_read_file(p))))
        except (OSError, ValueError):
            pass
    return result


def add_library_file(src: str) -> str:
    """
    Validates a library CSV and copies it into the user library folder.
    Returns the destination path. The shipped library is never modified.
    """
    _read_file(src)                       # raises ValueError if not a library file
    folder = user_library_dir()
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(dest):
        shutil.copyfile(src, dest)
    reload()
    return dest


def available() -> bool:
    """True if the bundled data file can be found."""
    return os.path.exists(default_path())


def _key(name: str) -> str:
    return re.sub(r"[\s\-,()\[\]]", "", str(name).lower().replace("α", "a-"))


@functools.lru_cache(maxsize=4)
def _monomer_index(path: Optional[str] = None) -> dict:
    """{canonical name: (abbreviation, number of entries)}."""
    idx = {}
    for e in load(path):
        for name, abbr in ((e.monomer_1, e.abbr_1), (e.monomer_2, e.abbr_2)):
            if not name:
                continue
            a, n = idx.get(name, ("", 0))
            idx[name] = (a or abbr, n + 1)
    return idx


def monomers(path: Optional[str] = None) -> list:
    """All monomer names in the library, most-documented first."""
    idx = _monomer_index(path)
    return sorted(idx, key=lambda n: (-idx[n][1], n.lower()))


def abbreviation(name: str, path: Optional[str] = None) -> str:
    return _monomer_index(path).get(name, ("", 0))[0]


def display_name(name: str, path: Optional[str] = None) -> str:
    abbr = abbreviation(name, path)
    return f"{name} ({abbr})" if abbr else name


# Common alternative names typed by users -> library monomer names.
_ALIASES = {
    "1,3-butadiene": "Butadiene", "buta-1,3-diene": "Butadiene", "bd": "Butadiene",
    "vinylbenzene": "Styrene", "ethenylbenzene": "Styrene",
    "propenenitrile": "Acrylonitrile", "vinyl cyanide": "Acrylonitrile",
    "furan-2,5-dione": "Maleic anhydride",
}


def resolve(query: str, path: Optional[str] = None) -> Optional[str]:
    """
    Maps user text (name, abbreviation, or 'Name (ABBR)') onto a library
    monomer name, or None if nothing matches exactly.
    """
    if not query:
        return None
    q = re.sub(r"\s*\([^()]*\)\s*$", "", query.strip())
    idx = _monomer_index(path)
    if q in idx:
        return q
    k = _key(q)
    for name, (abbr, _) in idx.items():
        if _key(name) == k or (abbr and abbr.lower() == q.lower()):
            return name
    alias = _ALIASES.get(q.lower())
    return alias if alias in idx else None


def find_monomers(text: str, limit: int = 50, path: Optional[str] = None) -> list:
    """Monomer names whose name or abbreviation contains ``text``."""
    k = _key(text)
    idx = _monomer_index(path)
    hits = [n for n, (a, _) in idx.items() if k in _key(n) or (a and k == _key(a))]
    hits.sort(key=lambda n: (not _key(n).startswith(k), -idx[n][1], n.lower()))
    return hits[:limit]


def search(text: str = "", usable_only: bool = False, datasets=None,
           path: Optional[str] = None) -> list:
    """Entries where either monomer (name or abbreviation) contains ``text``."""
    k = _key(text)
    out = []
    for e in load(path):
        if usable_only and not e.usable:
            continue
        if datasets and e.dataset not in datasets:
            continue
        if k and not any(k in _key(s) for s in (e.monomer_1, e.monomer_2, e.abbr_1, e.abbr_2)):
            continue
        out.append(e)
    return out


def pair_entries(monomer_a: str, monomer_b: str, usable_only: bool = True,
                 path: Optional[str] = None) -> list:
    """
    All determinations for the pair, oriented so ``r_a`` belongs to
    ``monomer_a``. Sorted with the proposed default first.
    """
    out = []
    for e in load(path):
        if usable_only and not e.usable:
            continue
        if e.r1 is None or e.r2 is None:
            continue
        if e.monomer_1 == monomer_a and e.monomer_2 == monomer_b:
            out.append(PairEntry(e, monomer_a, monomer_b, e.r1, e.r2))
        elif e.monomer_1 == monomer_b and e.monomer_2 == monomer_a:
            out.append(PairEntry(e, monomer_a, monomer_b, e.r2, e.r1))
    out.sort(key=_preference)
    return out


def _relative_ci(e: Entry) -> float:
    """Mean relative 95% interval of r1 and r2; inf when not reported."""
    if e.r1_ci95 is None or e.r2_ci95 is None or not e.r1 or not e.r2:
        return float("inf")
    return (e.r1_ci95 / e.r1 + e.r2_ci95 / e.r2) / 2.0


def _preference(pe: PairEntry):
    e = pe.entry
    t = e.temperature_value()
    return (
        not e.usable,
        any(f in SOFT_FLAGS for f in e.flags),
        DATASET_PRIORITY.get(e.dataset, 9),
        _relative_ci(e),                     # most precise determination first
        abs(t - REFERENCE_TEMPERATURE_C) if t is not None else 1e3,
        not bool(e.doi),
        e.id,
    )


def default_entry(monomer_a: str, monomer_b: str, path: Optional[str] = None):
    """
    The entry proposed by default for a pair, or None.

    Policy (documented in docs/REACTIVITY_LIBRARY.md): usable rows only;
    rows without quality flags before flagged ones; then the narrowest
    reported 95% intervals (relative to r), then rows with a DOI.
    It is a starting point, not a recommendation of the 'best' value.
    """
    entries = pair_entries(monomer_a, monomer_b, path=path)
    return entries[0] if entries else None


def get_entry(entry_id: str, path: Optional[str] = None) -> Optional[Entry]:
    for e in load(path):
        if e.id == entry_id:
            return e
    return None


def build_r_matrix(monomer_names: list, choices: Optional[dict] = None,
                   path: Optional[str] = None):
    """
    Builds the engine's ``R_values`` (row i = [r_ij for j != i]) for an
    n-monomer system using the pairwise terminal-model approximation.

    Args:
        monomer_names: library names, in engine order (monomer 1 first).
        choices: optional {(i, j): entry_id} overriding the default entry
                 for the pair of positions i < j.

    Returns:
        (R_values, sources, missing) where ``sources`` maps "i-j" (1-based)
        to the entry id used and ``missing`` lists 1-based (i, j) pairs with
        no usable entry (their r values are returned as None).
    """
    n = len(monomer_names)
    choices = choices or {}
    rmat = [[None] * n for _ in range(n)]
    sources, missing = {}, []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = monomer_names[i], monomer_names[j]
            pe = None
            wanted = choices.get((i, j))
            if wanted:
                pe = next((p for p in pair_entries(a, b, usable_only=False, path=path)
                           if p.entry.id == wanted), None)
            if pe is None:
                pe = default_entry(a, b, path=path)
            if pe is None:
                missing.append((i + 1, j + 1))
                continue
            rmat[i][j], rmat[j][i] = pe.r_a, pe.r_b
            sources[f"{i + 1}-{j + 1}"] = pe.entry.id
    r_values = [[rmat[i][j] for j in range(n) if j != i] for i in range(n)]
    return r_values, sources, missing
