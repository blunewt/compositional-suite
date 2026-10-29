import csv
import os
import subprocess
import sys

import pytest

from compositional_suite import reactivity_library as rl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_data_file_is_consistent():
    entries = rl.load()
    assert len(entries) > 100
    assert {e.dataset for e in entries} == {"own_curated"}        # the only shipped list
    assert len({e.id for e in entries}) == len(entries)
    for e in entries:
        if e.usable:
            assert e.r1 is not None and e.r2 is not None and e.r1 >= 0 and e.r2 >= 0
            assert e.monomer_1 and e.monomer_2 and e.model == "terminal"
        assert e.dataset in rl.DATASET_PRIORITY


def test_orientation_is_swapped_for_reversed_lookup():
    ab = rl.pair_entries("Styrene", "Methyl methacrylate")
    ba = rl.pair_entries("Methyl methacrylate", "Styrene")
    by_id = {p.entry.id: p for p in ba}
    for p in ab:
        q = by_id[p.entry.id]
        assert (p.r_a, p.r_b) == (q.r_b, q.r_a)


def test_default_prefers_unflagged_precise_determinations(local_dir, tmp_path_factory):
    # Synthetic test pair (not real data): one flagged, one imprecise, one precise entry.
    src = tmp_path_factory.mktemp("src") / "pair.csv"
    with open(src, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "monomer_1", "monomer_2", "r1", "r1_ci95", "r2", "r2_ci95",
                    "usable", "flags", "dataset"])
        w.writerow(["T-flagged", "Testmonomer A", "Testmonomer B", "1.0", "0.01", "1.0", "0.01",
                    "yes", "atypical_value_check_source", "own_curated"])
        w.writerow(["T-wide", "Testmonomer A", "Testmonomer B", "1.0", "0.5", "1.0", "0.5",
                    "yes", "", "own_curated"])
        w.writerow(["T-narrow", "Testmonomer A", "Testmonomer B", "1.0", "0.05", "1.0", "0.05",
                    "yes", "", "own_curated"])
    rl.add_library_file(str(src))
    d = rl.default_entry("Testmonomer A", "Testmonomer B")
    assert d.entry.id == "T-narrow"


def test_resolve_accepts_names_and_abbreviations():
    assert rl.resolve("MMA") == "Methyl methacrylate"
    assert rl.resolve("Styrene (STY)") == "Styrene"
    assert rl.resolve("n-butyl acrylate") == "n-Butyl acrylate"
    assert rl.resolve("definitely-not-a-monomer") is None


def test_build_r_matrix_layout_and_missing_pairs():
    R, src, missing = rl.build_r_matrix(["Styrene", "Methyl methacrylate", "Ethylene"])
    e12 = rl.pair_entries("Styrene", "Methyl methacrylate")[0]
    assert R[0][0] == pytest.approx(e12.r_a) and R[1][0] == pytest.approx(e12.r_b)
    assert src["1-2"] == e12.entry.id
    for i, j in missing:                       # pairs without data stay unset
        assert R[i - 1][j - 2] is None


@pytest.fixture
def local_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPOSITIONAL_SUITE_LIBRARY_DIR", str(tmp_path))
    rl.reload()
    yield tmp_path
    monkeypatch.undo()
    rl.reload()


def test_local_addon_is_merged_and_validated(local_dir, tmp_path_factory):
    src = tmp_path_factory.mktemp("src") / "extra.csv"
    with open(src, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "monomer_1", "abbr_1", "monomer_2", "abbr_2", "r1", "r2", "usable", "dataset"])
        w.writerow(["X-1", "Styrene", "STY", "Vinyl acetate", "VAc", "55", "0.01", "yes", "own_curated"])
    shipped = len(rl.load())
    rl.add_library_file(str(src))
    assert len(rl.load()) == shipped + 1
    added = [p for p in rl.pair_entries("Vinyl acetate", "Styrene") if p.entry.id == "X-1"]
    assert [(p.r_a, p.r_b) for p in added] == [(0.01, 55.0)]      # re-oriented
    bad = tmp_path_factory.mktemp("bad") / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        rl.add_library_file(str(bad))


def test_common_alternative_names_resolve():
    assert rl.resolve("furan-2,5-dione") == "Maleic anhydride"
    assert rl.resolve("vinylbenzene") == "Styrene"
