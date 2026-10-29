import json
import os

import pytest

from compositional_suite import recipe_io as rio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def test_legacy_ctp_recipe_migrates():
    r = rio.load_recipe(os.path.join(HERE, "fixtures", "legacy_engine1_ctp_recipe.json"))
    assert r["compositional_suite_recipe"] == rio.RECIPE_FORMAT_VERSION
    assert r["global"]["ctr"] == "ideal"            # ctp = 1.0 meant one monomer per step
    assert r["_migration_notes"]


def test_legacy_ctp_mapping_is_exact():
    old = {"global": {"n_monomers": 2, "n_chains": 10, "r_values": [[1], [1]], "ctp": 0.8},
           "blocks": [{"mrs": [1, 1], "dp": 50, "conversion": 90}]}
    new, notes = rio.migrate_recipe(old)
    assert new["global"]["ctr"] == pytest.approx(50 * 0.8 / 0.2)   # BLOCK_LEN*CTP/(1-CTP)
    assert "ctp" not in new["global"] and notes


def test_example_recipes_load():
    folder = os.path.join(ROOT, "examples", "recipes")
    for name in os.listdir(folder):
        if name.endswith(".json"):
            rio.load_recipe(os.path.join(folder, name))


def test_newer_format_is_refused():
    with pytest.raises(ValueError, match="newer"):
        rio.migrate_recipe({"compositional_suite_recipe": 99, "global": {}})


@pytest.mark.parametrize("bad", [[[-0.1], [1.0]], [["x"], [1.0]], [[1.0, 2.0], [1.0]]])
def test_bad_r_values_rejected(bad):
    r = {"global": {"n_monomers": 2, "n_chains": 10, "r_values": bad, "ctr": 5},
         "blocks": [{"mrs": [1, 1], "dp": 5, "conversion": 50}]}
    with pytest.raises(ValueError):
        rio.validate_recipe(r)


def test_save_load_round_trip(tmp_path):
    r = rio.build_recipe({"n_monomers": 2, "n_chains": 50, "r_values": [[0.5], [2.0]],
                          "ctr": [10.0, float("inf")], "prune_dp": 0,
                          "monomer_names": ["Styrene", "Methyl methacrylate"],
                          "r_sources": {"1-2": "OWN-0003"}},
                         [{"mrs": [50, 50], "dp": 20, "conversion": 90, "purge": False}])
    path = tmp_path / "r.json"
    rio.save_recipe(str(path), dict(r, _migration_notes=["x"]))
    assert "_migration_notes" not in json.loads(path.read_text())
    back = rio.load_recipe(str(path))
    assert back["global"]["ctr"] == [10.0, "ideal"]
    assert back["global"]["monomer_names"] == ["Styrene", "Methyl methacrylate"]


def test_library_r_values_resolve():
    r = rio.validate_recipe({
        "global": {"n_monomers": 2, "n_chains": 10, "r_values": "library",
                   "monomer_names": ["STY", "MMA"], "ctr": 10},
        "blocks": [{"mrs": [1, 1], "dp": 5, "conversion": 50}]})
    g = rio.resolve_library_r_values(r)["global"]
    assert len(g["r_values"]) == 2 and g["r_sources"]["1-2"]

