import os

from compositional_suite import headless, recipe_io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_cli_end_to_end(tmp_path):
    rc = headless.main(["--config", os.path.join(ROOT, "examples", "recipes",
                                                 "sample_random_copolymer.json"),
                        "--out", str(tmp_path), "--seed", "1", "--quiet"])
    assert rc == 0
    chains = recipe_io.read_chains_csv(str(tmp_path / "chains.csv"))
    assert len(chains) == 5000
    meta = (tmp_path / "chains.meta.json").read_text()
    assert '"status": "ok"' in meta and '"engine_version"' in meta


def test_cli_rejects_bad_recipe(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"global": {"n_monomers": 2}}')
    assert headless.main(["--config", str(bad), "--out", str(tmp_path)]) == 2
