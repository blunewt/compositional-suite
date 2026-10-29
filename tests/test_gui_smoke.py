"""
Off-screen smoke test of the desktop application (QT_QPA_PLATFORM=offscreen).

Drives the real window: two Simulator blocks, export, the auto-updated
Analysis tab, the reactivity-ratio library dialogs, a small Batch scan and a
descriptor-space load. Modal dialogs are stubbed so nothing waits for a click.
"""
import json
import time

import pytest

pytest.importorskip("PyQt5.QtWidgets")
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(app, monkeypatch):
    yes = staticmethod(lambda *a, **k: QMessageBox.Yes)
    for name in ("warning", "information", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, yes)
    from compositional_suite.app import CompositionalSuiteApp
    w = CompositionalSuiteApp()
    yield w
    w.close()


def _wait(app, cond, timeout=120):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.01)
        assert time.time() - t0 < timeout, "timed out"
    app.processEvents()


def _run_block(app, win, n_before):
    win.run_block()
    _wait(app, lambda: win.engine is not None and win.engine.block_num > n_before
          and win._sim_worker is not None and not win._sim_worker.isRunning()
          and win.polymerize_btn.isEnabled())


def test_simulator_export_and_analysis(app, win, monkeypatch, tmp_path):
    win.n_chains_spin.setValue(200)
    win.block_len_spin.setValue(30)
    _run_block(app, win, 0)
    # synthesis-level inputs are locked once chains exist
    assert not win.n_mons_spin.isEnabled() and not win.rr_group_box.isEnabled()
    win.mr_inputs[0].setValue(10.0)
    win.mr_inputs[1].setValue(90.0)
    _run_block(app, win, 1)
    assert [b["dp"] for b in win.engine.block_history] == [30, 30]
    # every finished block was added to the 3D Descriptor Space automatically
    runs = win.descriptor_space_data.get("Group A", {})
    assert len(runs) == 2                                   # one group per synthesis
    first, second = list(runs)
    assert first.startswith("r 1/1") and "feed" in first     # named by r and feed
    assert second.startswith(first) and "→" in second       # block 2 adds its feed

    # a widget changed after the run must not leak into the export
    win.r_inputs[0][0].setValue(9.9)
    out = tmp_path / "chains.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    win.export_data()
    meta = json.loads((tmp_path / "chains.meta.json").read_text())
    assert len(meta["blocks"]) == 2 and meta["blocks"][1]["mrs"] == [10.0, 90.0]
    assert meta["r_ij"] == [[1.0], [1.0]]

    # Analysis tab refreshed automatically, metrics include SD
    _wait(app, lambda: win._analysis_worker is None and win.metrics_display.toPlainText())
    text = win.metrics_display.toPlainText()
    assert "Structural Deviation" in text and "Best-matching gradient fraction" in text

    # recipe saved after a run reproduces the whole synthesis
    recipe = win._gather_simulator_recipe()
    assert len(recipe["blocks"]) == 2 and recipe["ui"]["next_block"]

    # the 3D Descriptor Space accepts the exported file
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([str(out)], "")))
    win.load_csv_for_descriptor_space()
    assert any(m["filename"] == "chains.csv" for m in win.descriptor_space_metrics)

    # descriptor-space coordinates export as CSV
    cs_csv = tmp_path / "cs.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(cs_csv), "")))
    win.export_descriptor_space_csv()
    rows = cs_csv.read_text().splitlines()
    assert rows[0] == "group,file,kappa_star,mean_run_length_L,U", rows
    assert len(rows) == 1 + len(win.descriptor_space_metrics) and any("chains.csv" in r for r in rows)

    # SVG exports carry provenance metadata
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path)))
    win.export_descriptor_space_plots()
    svg = (tmp_path / "Descriptor_Space.svg").read_text()
    assert "Compositional Suite" in svg and '"engine_version"' in svg and '"coordinates"' in svg
    sim_svg = tmp_path / "sim.svg"
    win.save_ax_as_svg(win.ax_consumption, str(sim_svg))
    text = sim_svg.read_text()
    assert "<dc:description>" in text and '"blocks"' in text and '"synthesis"' in text


def test_library_fill_and_browser(app, win, monkeypatch):
    from compositional_suite import library_dialogs, reactivity_library as rl

    def fake_exec(dlg):
        for combo, name in zip(dlg.name_combos, ["Styrene", "Methyl methacrylate"]):
            combo.setEditText(name)
        dlg.lookup()
        dlg._accept()
        return QDialog.Accepted

    monkeypatch.setattr(library_dialogs.FillFromLibraryDialog, "exec_", fake_exec)
    win.fill_r_from_library("sim")
    default = rl.default_entry("Styrene", "Methyl methacrylate")
    assert win.r_inputs[0][0].value() == pytest.approx(default.r_a)
    assert win.r_inputs[1][0].value() == pytest.approx(default.r_b)
    setup = win._simulator_global_setup()
    assert setup["monomer_names"] == ["Styrene", "Methyl methacrylate"]
    assert setup["r_sources"] == {"1-2": default.entry.id}
    win.r_inputs[0][0].setValue(0.3)                     # hand edit drops the claim
    assert "r_sources" not in win._simulator_global_setup()

    browser = library_dialogs.LibraryBrowserDialog(win, initial_text="MMA")
    assert browser.table.rowCount() > 5
    browser.close()


def test_batch_scan_sequential(app, win, tmp_path):
    win.n_chains_spin_batch.setValue(80)
    win.num_steps_spin.setValue(2)
    win.parallel_check.setChecked(False)
    win.output_dir_path = str(tmp_path)
    win.run_batch_simulation()
    assert len(win.batch_results) == 2
    assert len(list(tmp_path.glob("*.csv"))) == 2


def test_sd_metrics_only_for_two_monomers():
    from compositional_suite.app import _compute_sd_for_ensemble
    assert "evaluation" in _compute_sd_for_ensemble([[1, 2, 1, 2, 2]] * 30)
    out = _compute_sd_for_ensemble([[1, 2, 3, 1, 2]] * 30)
    assert "two-monomer" in out["error"]


def test_cone_hidden_for_three_monomers():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from compositional_suite import plotting
    fig = Figure()
    axes = (fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133))
    plotting.plot_2d_kde_with_marginals(axes, [[1, 2, 3, 1, 2]] * 30, 3)
    assert not any(ax.get_visible() for ax in axes)
    plotting.plot_2d_kde_with_marginals(axes, [[1, 2, 1, 2, 2, 1]] * 30, 2)
    assert all(ax.get_visible() for ax in axes)


def test_target_dispersity_is_the_default_mode(app, win):
    assert win.target_pdi_check.isChecked() and win.target_pdi_spin.isEnabled()
    assert not win.ctr_group_box.isEnabled()
    assert win.target_pdi_spin.value() == pytest.approx(1.20)
    win.target_pdi_check.setChecked(False)
    win.reset_simulation()
    assert win.target_pdi_check.isChecked()


def test_descriptor_space_rename_regroup_and_export(app, win, monkeypatch, tmp_path):
    import zipfile
    assert win.tabs.tabText(win.tabs.indexOf(win.descriptor_space_tab)) == "3D Descriptor Space"
    win.clear_descriptor_space_list()
    win._cs_add_entry("G1", "a", [[1, 2] * 20] * 30, {"source": "test", "meta": None})
    win._cs_add_entry("G1", "b", [[1, 1, 2, 2] * 10] * 30, {"source": "test", "meta": None})
    win._cs_refresh_table()
    win.run_descriptor_space_analysis()
    win.cs_file_list.item(0, 0).setText("alternating")            # double-click rename
    assert set(win.descriptor_space_data["G1"]) == {"alternating", "b"}
    row = next(r for r in range(win.cs_file_list.rowCount())
               if win.cs_file_list.item(r, 0).text() == "b")
    win.cs_file_list.item(row, 1).setText("G2")                    # double-click regroup
    assert "b" in win.descriptor_space_data["G2"] and "b" not in win.descriptor_space_data["G1"]
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "all.zip"), "")))
    win.export_descriptor_space_all()
    names = zipfile.ZipFile(tmp_path / "all.zip").namelist()
    assert {"manifest.json", "descriptor_coordinates.csv", "G1/alternating.csv", "G2/b.csv"} <= set(names)


def test_default_groups_are_alphabetical(app, win):
    win.clear_descriptor_space_list()
    assert win._cs_next_default_group() == "Group A"
    for g in ("Group C", "Group A"):
        win._cs_add_entry(g, "x", [[1, 2, 1]] * 5, {})
    assert win._cs_next_default_group() == "Group B"
    win._cs_refresh_table()
    groups = [win.cs_file_list.item(r, 1).text() for r in range(win.cs_file_list.rowCount())]
    assert groups == sorted(groups)


def test_up_to_eight_monomers(app, win, monkeypatch):
    from compositional_suite import plotting
    assert win.n_mons_spin.maximum() == 8 and win.n_mons_spin_batch.maximum() == 8
    assert len(set(plotting.MONOMER_COLORS[:8])) == 8          # eight distinct colours
    win.n_mons_spin.setValue(8)
    assert len(win.r_inputs) == 8 and len(win.mr_inputs) == 8
    # a recipe beyond the interface limit is refused with a clear message, not clamped
    recipe = {"compositional_suite_recipe": 2,
              "global": {"n_monomers": 9, "n_chains": 10, "r_values": [[1] * 8] * 9, "ctr": "ideal"},
              "blocks": [{"mrs": [1] * 9, "dp": 5, "conversion": 50}]}
    with pytest.raises(ValueError, match="supports 2–8"):
        win._apply_simulator_recipe(recipe)


def test_program_has_an_icon(app, win):
    import os
    import re
    from compositional_suite import app as app_module
    assert not win.windowIcon().isNull()                      # window / taskbar icon
    assert not app_module.app_icon().isNull()
    # the icon file must travel with pip installs and frozen builds
    root = os.path.dirname(os.path.dirname(os.path.abspath(app_module.__file__)))
    assert "data/*.png" in open(os.path.join(root, "pyproject.toml"), encoding="utf-8").read()
    # every icon the build workflow embeds must exist in the repository
    wf = open(os.path.join(root, ".github", "workflows", "build.yml"), encoding="utf-8").read()
    icons = re.findall(r"--icon\s+(\S+)", wf)
    assert len(icons) == 2
    for rel in icons:
        assert os.path.exists(os.path.join(root, rel)), rel


def test_about_dialog_states_dtu_copyright(app, win):
    text = win.about_text()
    assert "Copyright \u00a9 2026 Technical University of Denmark" in text
    assert "Associate Professor" in text and "Department of Health Technology" in text


def test_taskbar_identity_only_when_running_from_source(monkeypatch):
    """The packaged .exe must not set a taskbar ID (Windows uses the exe's own icon)."""
    import ctypes
    import sys
    from compositional_suite import app as app_module
    calls = []
    shell32 = type("S", (), {"SetCurrentProcessExplicitAppUserModelID": staticmethod(lambda i: calls.append(i))})
    monkeypatch.setattr(ctypes, "windll", type("W", (), {"shell32": shell32})(), raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert app_module._set_windows_app_id() is False and calls == []          # packaged .exe
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert app_module._set_windows_app_id() is True and calls == [app_module.WINDOWS_APP_ID]   # from source
