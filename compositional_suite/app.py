"""
app.py
-------------------
PyQt5 desktop application for stochastic copolymerisation simulation and
analysis.

Application structure
~~~~~~~~~~~~~~~~~~~~~
The main window (``CompositionalSuiteApp``) contains four tabs:

1. **Simulator** – configure reactivity ratios, monomer feed, and Ctr; run
   single or multi-block polymerisations; inspect conversion curves and
   sequence heatmaps in real time.
2. **Analysis** – load a previously simulated or externally generated CSV
   ensemble; visualise positional, compositional, and sequence-level metrics.
3. **3D Descriptor Space** – overlay multiple ensembles in a 3-D
   (κ*, ⟨L⟩, U) descriptor space; rename and regroup runs; export them together.
4. **Batch Simulator** – sweep one or two parameters across a grid, running a
   full simulation at each point and collecting summary metrics for export.

Background threading
~~~~~~~~~~~~~~~~~~~~
Long-running operations (``SimulationEngine`` propagation and heavy analysis
passes) are executed on ``QThread`` workers so the UI remains responsive.
``_SimWorker`` and ``_AnalysisWorker`` emit ``finished`` / ``error`` signals
back to the main thread when complete.

Entry point
~~~~~~~~~~~
Launch with ``compositional_suite`` (installed) or ``python run_compositional_suite.py`` from a
checkout. This file cannot be run directly: it uses package-relative imports.
"""

import sys
import json
import os
import csv
import math
import logging
import datetime
import multiprocessing
import numpy as np
import pandas as pd
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QSpinBox, QDoubleSpinBox, QTextEdit,
    QFileDialog, QMessageBox, QLineEdit, QTabWidget, QCheckBox, QComboBox,
    QListWidget, QProgressBar, QDialog, QDialogButtonBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QAction,
    QScrollArea, QFrame, QShortcut,
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QKeySequence
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

plt.rcParams['font.family'] = ['DejaVu Sans', 'Segoe UI', 'Arial', 'sans-serif']

from . import analysis
from . import simulation_engine
from . import plotting
from . import recipe_io
from . import headless
from . import reactivity_library
from . import __version__, __url__, __authors__, __copyright__, __developed_by__

try:
    from . import sd_metrics as _sd_metrics
    _SD_METRICS_AVAILABLE = True
except ImportError:
    _sd_metrics = None
    _SD_METRICS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants 
# ---------------------------------------------------------------------------
# No lower bound beyond 1 — small ensembles are useful for quick exploration
# and for inspecting individual chains. Distribution statistics (SD sigma,
# skewness, CV) are noisy below a few thousand chains; the validation suite
# uses 5000.
_MIN_CHAINS = 1
_MAX_CHAINS = 100_000
_DEFAULT_CHAINS = 1_000
_MIN_MONOMERS = 2
_MAX_MONOMERS = 8      # the plotting palette has 8 distinct monomer colours
_DEFAULT_BLOCK_LEN = 60
_DEFAULT_CONVERSION = 100.0
# Chain transfer constant k_tr/k_p. 10 is a moderate, chemically plausible
# default (well-matched trithiocarbonate territory).
_DEFAULT_CTR = 10.0
_MAX_CTR = 1e6               # spinbox ceiling; beyond this use "ideal exchange"
_DEFAULT_PDI_TARGET = 1.20   # default simulation mode: user-defined target dispersity
# Controls-column sizing. The column is re-fitted to its content whenever the
# monomer count changes, between a floor (2 monomers, uncramped) and a ceiling
# (beyond this the plots get squeezed, so a scrollbar is the lesser evil).
_CONTROLS_MIN_WIDTH = 470
_CONTROLS_MAX_WIDTH = 1100
_CONTROLS_PADDING = 8
# Monomer inputs (MR, Ctr) wrap onto a new row past this many, so the column
# grows downward (scrollable) rather than sideways (crowds the plots).
_INPUTS_PER_ROW = 2
# Ctr spans up to 1e6, so a spinbox sized to its widest value dominates the
# column. Cap the field width; long values still scroll within the field.
_MONOMER_SPIN_WIDTH = 95
# r_ij inputs: 4 decimals so published values such as 1e-4 do not round to 0,
# which the engine reads as "never homopropagates" (see check_r_values).
_R_DECIMALS = 4
_QWIDGETSIZE_MAX = 16777215   # Qt's "no maximum" sentinel
_BATCH_SCAN_MAX_STEPS = 100
_DESCRIPTOR_SPACE_KDE_ENTROPY_MAX = 5.0
_LAMBDA_AXIS_RANGE = (-1, 1)

# ---------------------------------------------------------------------------
# Application metadata (shown in Help > About / How to Cite)
# Edit these to match your release before publishing.
# ---------------------------------------------------------------------------
# Single source of truth: compositional_suite/__init__.py. Edit there, not here.
APP_NAME = "Compositional Suite"
APP_VERSION = __version__
APP_LICENSE = "MIT License"
APP_REPO_URL = __url__
APP_AUTHORS = __authors__

# Citation template. Replace the placeholders with the published reference
# (and DOI) once the Compositional Suite paper is out.
CITATION_PLAIN = (
    f"{APP_AUTHORS} ({{year}}). {APP_NAME}: a Monte-Carlo simulator for radical "
    "copolymerization sequence analysis (Version {version}) [Computer software]. "
    f"{APP_REPO_URL}"
)
CITATION_BIBTEX = (
    "@software{{compositional_suite_{year},\n"
    "  author  = {{{authors}}},\n"
    "  title   = {{{name}: a Monte-Carlo simulator for radical "
    "copolymerization sequence analysis}},\n"
    "  version = {{{version}}},\n"
    "  year    = {{{year}}},\n"
    "  url     = {{{url}}}\n"
    "}}"
)


# ---------------------------------------------------------------------------
# SD Metrics helper
# ---------------------------------------------------------------------------

def _compute_sd_for_ensemble(polymers, max_chains=10_000):
    """
    Structural deviation metrics (Conka et al.) for a two-monomer ensemble.

    Returns {'evaluation': sd_metrics.SDEvaluation} or {'error': message}.
    Like the binary cone of heterogeneity, SD is only defined for copolymers
    of exactly two monomers; other ensembles get an explanatory message.
    """
    if not polymers:
        return {'error': 'No polymers loaded'}
    try:
        return {'evaluation': _sd_metrics.evaluate(polymers, max_chains=max_chains)}
    except ValueError as exc:
        return {'error': str(exc)}


# ===========================================================================
# Background worker threads
# ===========================================================================

class _SimWorker(QThread):
    """
    Runs a single block polymerisation on a background thread so the UI stays
    responsive during long simulations.
    """
    finished = pyqtSignal(object, object, str)   # conv_hist, amounts_hist, status
    error = pyqtSignal(str)

    def __init__(self, engine, mrs, block_len, conversion, is_first_block,
                 purge_monomers=False, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._mrs = mrs
        self._block_len = block_len
        self._conversion = conversion
        self._is_first_block = is_first_block
        self._purge_monomers = purge_monomers

    def run(self):
        try:
            if self._is_first_block:
                result = self._engine.initialize_chains(
                    self._mrs, self._block_len, self._conversion
                )
            else:
                result = self._engine.add_new_block(
                    self._mrs, self._block_len, self._conversion,
                    purge_monomers=self._purge_monomers,
                )
            self.finished.emit(*result)
        except Exception as exc:
            self.error.emit(str(exc))


class _AnalysisWorker(QThread):
    """Runs calculate_and_display_all_metrics off the UI thread."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, polymers, parent=None):
        super().__init__(parent)
        self._polymers = polymers

    def run(self):
        try:
            p = self._polymers
            results = {
                'pdi': analysis.calculate_pdi(p),
                'kappa': analysis.calculate_kappa_metrics(p),
                'avg_comp': analysis.calculate_average_composition(p),
                'U': analysis.calculate_uncertainty_coefficient(p),
                'mv_spread': analysis.calculate_multivariate_spread(p),
                'spread': analysis.calculate_compositional_spread(p),
                'cmm': analysis.calculate_cmm(p),
                'grad': analysis.calculate_gradient_index(p),
                'run_length': analysis.calculate_mean_run_length(p),
                'all_mons': max((m for chain in p for m in chain), default=0),
                'n_chains': len(p),
                'sd': _compute_sd_for_ensemble(p) if _SD_METRICS_AVAILABLE else None,
            }
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


# ===========================================================================
# Main application
# ===========================================================================

class CompositionalSuiteApp(QMainWindow):
    """
    Main application window for Compositional Suite.

    Manages the top-level tab widget and coordinates state shared across tabs:
    the active ``SimulationEngine``, the currently loaded polymer ensemble,
    batch/descriptor-space results, and the last recorded conversion / amounts
    histories.

    UI is constructed entirely in ``__init__`` via the ``create_*`` helper
    methods; no .ui file is used.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Compositional Suite - Polymer Simulator & Analyzer")
        self.setWindowIcon(app_icon())
        self.setGeometry(100, 100, 1800, 950)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # State
        self.loaded_polymers = None
        self.loaded_filename = ""
        self.max_monomers_for_analysis = _MAX_MONOMERS
        self.last_conv_hist = None
        self.last_amounts_hist = None
        self.descriptor_space_data = {}
        self.descriptor_space_metrics = []
        # Per-ensemble info for the 3D Descriptor Space: {(group, name): {...}}
        #   source: file path or "simulator"; meta: provenance dict; desc: cached descriptors
        self.descriptor_space_info = {}
        self._cs_dirty = False
        self._sim_run_no = 0
        self._sim_run_group = None
        self.descriptor_space_artists_map = {}
        self.legend_map = {}
        self.group_color_map = {}
        self.output_dir_path = None
        self.batch_results = []
        self.last_analysis_data = None
        self.engine = None
        self.is_initialized = False
        self._sim_worker = None
        self._analysis_worker = None

        self.create_simulator_tab()
        self.create_analysis_tab()
        self.create_descriptor_space_tab()
        self.create_batch_tab()
        # Built last: the Export submenu and tab shortcuts reference tab widgets.
        self.create_menu_bar()
        # Default simulation mode: the user sets the target dispersity and Ctr
        # is solved from it (uncheck to enter Ctr directly or use ideal exchange).
        self.target_pdi_check.setChecked(True)

    # -----------------------------------------------------------------------
    # Menu bar
    # -----------------------------------------------------------------------

    # Per-tab action map. Keyed by tab attribute name; each entry gives the
    # primary "run" action and the list of (label, handler) export actions for
    # that tab. The File > Export submenu and the F5 / Ctrl+E shortcuts read
    # from this so the menu bar stays in sync with whichever tab is active.
    TAB_SHORTCUTS = [
        ("simulator_tab", "Simulator", "Ctrl+1"),
        ("analysis_tab", "Analysis", "Ctrl+2"),
        ("descriptor_space_tab", "3D Descriptor Space", "Ctrl+3"),
        ("batch_tab", "Batch Simulator", "Ctrl+4"),
    ]

    def create_menu_bar(self):
        """Builds the File, View and Help menus plus global keyboard shortcuts."""
        # ---- File ----
        file_menu = self.menuBar().addMenu("&File")

        # Recipe actions belong to the Simulator / Batch tabs; the Analysis and
        # 3D Descriptor Space tabs instead expose Load CSV. Visibility is toggled per tab
        # in _update_file_menu_actions().
        self.save_recipe_action = QAction("Save Recipe…", self)
        self.save_recipe_action.setShortcut("Ctrl+S")
        self.save_recipe_action.triggered.connect(self.save_recipe_from_menu)
        file_menu.addAction(self.save_recipe_action)

        self.load_recipe_action = QAction("Load Recipe…", self)
        self.load_recipe_action.setShortcut("Ctrl+O")
        self.load_recipe_action.triggered.connect(self.load_recipe_from_menu)
        file_menu.addAction(self.load_recipe_action)

        self.load_csv_action = QAction("Load CSV…", self)
        self.load_csv_action.setShortcut("Ctrl+O")
        self.load_csv_action.triggered.connect(self.load_csv_from_menu)
        file_menu.addAction(self.load_csv_action)

        file_menu.addSeparator()
        # Tab-aware: contents are rebuilt whenever the active tab changes.
        self.export_menu = file_menu.addMenu("Export")

        # ---- View (tab navigation) ----
        view_menu = self.menuBar().addMenu("&View")
        for idx, (_attr, name, key) in enumerate(self.TAB_SHORTCUTS):
            act = QAction(name, self)
            act.setShortcut(key)
            act.triggered.connect(
                lambda _checked=False, i=idx: self.tabs.setCurrentIndex(i)
            )
            view_menu.addAction(act)

        # ---- Help ----
        help_menu = self.menuBar().addMenu("&Help")

        assumptions_action = QAction("Model Assumptions…", self)
        assumptions_action.triggered.connect(self.show_model_assumptions)
        help_menu.addAction(assumptions_action)

        ratios_action = QAction("Reactivity Ratio Library…", self)
        ratios_action.triggered.connect(self.show_reactivity_library)
        help_menu.addAction(ratios_action)

        help_menu.addSeparator()
        shortcuts_action = QAction("Keyboard Shortcuts…", self)
        shortcuts_action.triggered.connect(self.show_keyboard_shortcuts)
        help_menu.addAction(shortcuts_action)

        docs_action = QAction("Documentation", self)
        docs_action.setShortcut("F1")
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)

        help_menu.addSeparator()
        cite_action = QAction("How to Cite…", self)
        cite_action.triggered.connect(self.show_how_to_cite)
        help_menu.addAction(cite_action)

        about_action = QAction("About…", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        # ---- Global context-aware shortcuts ----
        # Ctrl+Return runs the active tab's primary action. Ctrl+E (primary
        # export) is carried by the first Export submenu action, set per tab in
        # _rebuild_export_menu so it always targets the active tab.
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._run_current_tab)
        QShortcut(QKeySequence("Ctrl+Enter"), self, activated=self._run_current_tab)
        # Ctrl+R resets (Simulator) or clears all data (Descriptor Space).
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._reset_current_tab)
        # Ctrl+T transfers the simulated ensemble to the Analysis tab.
        QShortcut(QKeySequence("Ctrl+T"), self, activated=self._transfer_current_tab)

        # Keep the Export submenu and File-menu actions in sync with the tab.
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._on_tab_changed()

    # -----------------------------------------------------------------------
    # Tab-aware menu / shortcut helpers
    # -----------------------------------------------------------------------

    def _current_tab_exports(self):
        """Returns [(label, handler), ...] of export actions for the active tab."""
        w = self.tabs.currentWidget()
        if w is self.simulator_tab:
            return [("Sequences (CSV)…", self.export_data),
                    ("Plots (SVG)…", self.export_plots)]
        if w is self.analysis_tab:
            return [("Plots (SVG)…", self.export_analysis_plots),
                    ("Data (CSV)…", self.export_analysis_csv)]
        if w is self.descriptor_space_tab:
            return [("Plots (SVG)…", self.export_descriptor_space_plots),
                    ("Coordinates (CSV)…", self.export_descriptor_space_csv),
                    ("All ensembles (ZIP)…", self.export_descriptor_space_all),
                    ("Interactive 3D (HTML)…", self.export_descriptor_space_html)]
        if w is self.batch_tab:
            return [("Plot Data (XLS)…", self.export_batch_plot_data)]
        return []

    def _rebuild_export_menu(self):
        """Repopulates File > Export with the active tab's export actions."""
        self.export_menu.clear()
        exports = self._current_tab_exports()
        if not exports:
            placeholder = self.export_menu.addAction("Nothing to export here")
            placeholder.setEnabled(False)
            return
        for i, (label, handler) in enumerate(exports):
            act = self.export_menu.addAction(label)
            if i == 0:
                act.setShortcut("Ctrl+E")  # primary export for this tab
            act.triggered.connect(handler)

    def _on_tab_changed(self, *args):
        """Keeps tab-dependent menu entries in sync with the active tab."""
        self._rebuild_export_menu()
        self._update_file_menu_actions()
        if (getattr(self, "descriptor_space_tab", None) is not None
                and self.tabs.currentWidget() is self.descriptor_space_tab
                and self._cs_dirty and self.descriptor_space_data):
            self.run_descriptor_space_analysis()

    def _update_file_menu_actions(self):
        """
        Shows recipe Save/Load on the Simulator & Batch tabs, and Load CSV on the
        Analysis & 3D Descriptor Space tabs. Inapplicable actions are hidden *and* disabled
        so their shortcuts (Ctrl+O is shared) never fire on the wrong tab.
        """
        w = self.tabs.currentWidget()
        is_recipe_tab = w in (self.simulator_tab, self.batch_tab)
        is_csv_tab = w in (self.analysis_tab, self.descriptor_space_tab)
        for act in (self.save_recipe_action, self.load_recipe_action):
            act.setVisible(is_recipe_tab)
            act.setEnabled(is_recipe_tab)
        self.load_csv_action.setVisible(is_csv_tab)
        self.load_csv_action.setEnabled(is_csv_tab)

    def load_csv_from_menu(self):
        """File > Load CSV — routes to the active tab's CSV loader."""
        w = self.tabs.currentWidget()
        if w is self.analysis_tab:
            self.load_csv_file()
        elif w is self.descriptor_space_tab:
            self.load_csv_for_descriptor_space()

    def _reset_current_tab(self):
        """Ctrl+R — resets the Simulator, or clears all descriptor-space data."""
        w = self.tabs.currentWidget()
        if w is self.simulator_tab:
            self.reset_simulation()
        elif w is self.descriptor_space_tab:
            self.clear_descriptor_space_list()

    def _transfer_current_tab(self):
        """Ctrl+T — transfers the simulated ensemble to the Analysis tab."""
        if self.tabs.currentWidget() is self.simulator_tab:
            self.transfer_sim_to_analysis()

    def _run_current_tab(self):
        """Ctrl+Enter — triggers the primary action of the active tab (if available)."""
        w = self.tabs.currentWidget()
        if w is self.simulator_tab and self.polymerize_btn.isEnabled():
            self.run_block()
        elif w is self.analysis_tab and self.loaded_polymers:
            self.calculate_and_display_all_metrics()
        elif w is self.descriptor_space_tab:
            self.run_descriptor_space_analysis()
        elif w is self.batch_tab and self.run_batch_btn.isEnabled():
            self.run_batch_simulation()

    def show_model_assumptions(self):
        """Displays the explicit modelling assumptions behind the engine."""
        text = (
            "<b>Compositional Suite — Model Assumptions</b>"
            "<p>Simulation results should be interpreted in light of the "
            "following assumptions built into the Monte Carlo engine:</p>"
            "<ol>"
            "<li><b>Linear chains.</b> No branching, crosslinking, or cyclisation; "
            "each chain is a 1-D sequence of monomer units.</li>"
            "<li><b>Terminal (first-order Markov) model.</b> Addition "
            "probabilities depend only on the identity of the terminal "
            "(chain-end) monomer via the reactivity ratios r_ij. Penultimate "
            "and longer-range effects are not modelled.</li>"
            "<li><b>No diffusion limitation.</b> Propagation is reaction-"
            "controlled throughout; no Trommsdorff/gel effect, and rate "
            "constants are independent of viscosity or conversion.</li>"
            "<li><b>Instantaneous initiation.</b> All chains are initiated at "
            "the start of a block and grow simultaneously; initiation kinetics "
            "and initiator efficiency are not represented.</li>"
            "<li><b>Ideal mixing.</b> The monomer pool is treated as perfectly "
            "homogeneous; every active chain end samples the same instantaneous "
            "feed composition.</li>"
            "</ol>"
            "<p>Dispersity arises from degenerative (RAFT) chain transfer, "
            "parameterised by the chain transfer constant "
            "<b>Ctr = k<sub>tr</sub>/k<sub>p</sub></b>. Monomers added per "
            "activation are geometric with mean 1 + ([M]/[CTA])/Ctr, so Đ narrows "
            "as exchange gets faster and as monomer is consumed.</p>"
            "<p><b>Scope: ideal living polymerization only.</b> All chains start "
            "together and none terminate; there is no radical generation, no "
            "initiator-derived or dead chains, no slow RAFT-agent consumption "
            "(hybrid behaviour) and no retardation. These non-ideal processes are "
            "outside the scope of Compositional Suite by design. Ctr therefore "
            "captures broadening from slow exchange only.</p>"
            "<p><b>Scope: linear chains only.</b> Every chain is one linear "
            "sequence; no reaction creates branch points or joins chains. "
            "Branched, hyperbranched and crosslinked polymers, networks, and "
            "cyclic or bottlebrush topologies are outside the scope, and divinyl "
            "crosslinkers are treated as ordinary monomers (which is misleading). "
            "For core-first stars and side chains grafted from a backbone, the "
            "simulated chains can represent individual arms or side chains, but "
            "not the whole molecule.</p>"
        )
        box = QMessageBox(self)
        box.setWindowTitle("Model Assumptions")
        box.setTextFormat(Qt.RichText)
        box.setText(text)
        box.setIcon(QMessageBox.Information)
        box.exec_()

    def show_reactivity_library(self):
        """Opens the searchable, read-only reactivity-ratio library browser."""
        if not reactivity_library.available():
            QMessageBox.warning(self, "Library not found",
                                "The reactivity-ratio data file is missing from "
                                "this installation (compositional_suite/data/).")
            return
        from .library_dialogs import LibraryBrowserDialog
        LibraryBrowserDialog(self).exec_()

    def show_keyboard_shortcuts(self):
        """Lists the application-wide keyboard shortcuts."""
        rows = [
            ("Ctrl+Enter", "Run — primary action of the active tab "),
            ("Ctrl+E", "Export"),
            ("Ctrl+S", "Save recipe"),
            ("Ctrl+O", "Load recipe or Load CSV"),
            ("Ctrl+R", "Reset"),
            ("Ctrl+T", "Jump to the Analysis tab (it already refreshes automatically after each block)"),
            ("Ctrl+1", "Go to Simulator tab"),
            ("Ctrl+2", "Go to Analysis tab"),
            ("Ctrl+3", "Go to 3D Descriptor Space tab"),
            ("Ctrl+4", "Go to Batch Simulator tab"),
            ("F1", "Open Documentation"),
        ]
        table = "".join(
            f"<tr><td style='padding:3px 18px 3px 0;'><b>{k}</b></td>"
            f"<td>{d}</td></tr>"
            for k, d in rows
        )
        box = QMessageBox(self)
        box.setWindowTitle("Keyboard Shortcuts")
        box.setTextFormat(Qt.RichText)
        box.setText(
            "<b>Keyboard Shortcuts</b>"
            "<p>Ctrl+Enter, Ctrl+E, Ctrl+R and Ctrl+T act on whichever tab is active.</p>"
            f"<table>{table}</table>"
        )
        box.exec_()

    def show_documentation(self):
        """Shows a concise built-in usage guide for Compositional Suite."""
        doc = (
            f"<h3>Compositional Suite {simulation_engine.ENGINE_VERSION}</h3>"
            "<p>Monte-Carlo simulator and sequence analyser for radical "
            "copolymerisation.</p>"

            "<h4>Tabs</h4>"
            "<ul>"
            "<li><b>Simulator</b> — set reactivity ratios, feed and Ctr, then "
            "<i>Polymerize</i> one or more blocks. Optionally goal-seek a target "
            "dispersity (Đ).</li>"
            "<li><b>Analysis</b> — PDI, composition, run-length, eigen and SD "
            "metrics. Updates automatically after every simulation block; you "
            "can also load a chain CSV directly.</li>"
            "<li><b>3D Descriptor Space</b> — map ensembles into (κ*, ⟨L⟩, U) "
            "space, comparable across numbers of monomers. Every Simulator block is "
            "added automatically; double-click a name or group to edit it; export "
            "SVG, CSV, all ensembles as one ZIP, or an interactive 3-D HTML.</li>"
            "<li><b>Batch Simulator</b> — sweep a monomer ratio across a block "
            "recipe; runs in parallel across CPU cores when enabled.</li>"
            "</ul>"

            "<h4>Recipes</h4>"
            "<p><b>File &gt; Save / Load Recipe</b> stores the active tab's setup "
            "as JSON (or YAML with PyYAML). Recipes are interchangeable with the "
            "headless CLI.</p>"

            "<h4>Exports &amp; provenance</h4>"
            "<p>Exported chain CSVs carry a commented metadata header plus a "
            "<code>.meta.json</code> sidecar recording engine version, reactivity "
            "ratios, feed, Ctr and block recipe — so a dataset is never orphaned "
            "from the conditions that produced it.</p>"

            "<h4>Headless / HPC use</h4>"
            "<p>Run without the GUI from any recipe file:</p>"
            "<pre>compositional-suite-cli --config exp.json --out ./run/ --seed 7</pre>"

            "<h4>More</h4>"
            "<h4>Reactivity-ratio library</h4>"
            "<p><b>From library…</b> (next to the reactivity ratios) fills r<sub>ij</sub> "
            "from published determinations and records which entries were used in "
            "recipes and exports. <b>Help &gt; Reactivity Ratio Library</b> browses "
            "every entry with its conditions and reference.</p>"
            "<p>See <b>Help &gt; Model Assumptions</b> for the modelling "
            "assumptions.</p>"
        )
        dialog = QDialog(self)
        dialog.setWindowTitle("Documentation")
        dialog.setMinimumSize(620, 560)
        layout = QVBoxLayout(dialog)
        view = QTextEdit()
        view.setReadOnly(True)
        view.setHtml(doc)
        layout.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec_()

    def show_how_to_cite(self):
        """Shows a citation template (plain + BibTeX) for Compositional Suite."""
        year = datetime.datetime.now().year
        plain = CITATION_PLAIN.format(year=year, version=APP_VERSION)
        bibtex = CITATION_BIBTEX.format(
            year=year, authors=APP_AUTHORS, name=APP_NAME,
            version=APP_VERSION, url=APP_REPO_URL,
        )

        dialog = QDialog(self)
        dialog.setWindowTitle("How to Cite")
        dialog.setMinimumSize(620, 420)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(
            "If Compositional Suite contributes to published work, please cite it. "
            "Update this template with the paper reference and DOI once available."
        ))
        layout.addWidget(QLabel("<b>Citation</b>"))
        plain_view = QTextEdit()
        plain_view.setReadOnly(True)
        plain_view.setMaximumHeight(90)
        plain_view.setPlainText(plain)
        layout.addWidget(plain_view)

        layout.addWidget(QLabel("<b>BibTeX</b>"))
        bib_view = QTextEdit()
        bib_view.setReadOnly(True)
        bib_view.setFontFamily("Courier New")
        bib_view.setPlainText(bibtex)
        layout.addWidget(bib_view)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec_()

    def about_text(self):
        """Rich text of the About dialog: versions, copyright, license, repository."""
        return (
            f"<h3>{APP_NAME} {APP_VERSION}</h3>"
            "<p>Monte-Carlo simulator and sequence analyser for radical "
            "copolymerisation.</p>"
            "<table>"
            f"<tr><td><b>Application version</b></td><td>&nbsp;{APP_VERSION}</td></tr>"
            f"<tr><td><b>Engine version</b></td>"
            f"<td>&nbsp;{simulation_engine.ENGINE_VERSION}</td></tr>"
            f"<tr><td><b>License</b></td><td>&nbsp;{APP_LICENSE}</td></tr>"
            f"<tr><td><b>Repository</b></td>"
            f"<td>&nbsp;<a href='{APP_REPO_URL}'>{APP_REPO_URL}</a></td></tr>"
            "</table>"
            f"<p>{__copyright__}<br>{__developed_by__}</p>"
            "<p style='color:#666;'>The engine version is stamped into every "
            "exported dataset's metadata for full provenance.</p>"
        )

    def show_about(self):
        """Shows the About dialog."""
        text = self.about_text()
        box = QMessageBox(self)
        box.setWindowTitle("About Compositional Suite")
        box.setTextFormat(Qt.RichText)
        box.setText(text)
        box.setIcon(QMessageBox.Information)
        box.exec_()

    # -----------------------------------------------------------------------
    # Tab builders
    # -----------------------------------------------------------------------

    def _wrap_in_scroll(self, panel: QWidget, width: int = None) -> QScrollArea:
        """
        Puts a controls panel inside a vertical scroll area.

        The left-hand control columns are taller than most windows; scrolling
        keeps every group at its natural height (no squished labels) and ensures
        the lower controls are always reachable.

        The width is not fixed: the reactivity-ratio and Ctr grids grow with the
        monomer count, so it is re-fitted to the content by
        ``_fit_controls_width`` whenever that count changes. The panel keeps a
        reference to its scroll area so the two can be re-fitted together.
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setFrameShape(QFrame.NoFrame)
        # As-needed rather than always-off: if the content ever exceeds the
        # width cap, a scrollbar is far better than silently clipping controls.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Start at the floor so the column has a sane width before the first
        # deferred fit lands; that fit then adjusts it to the real content.
        scroll.setFixedWidth(_CONTROLS_MIN_WIDTH if width is None else width)
        self._fit_controls_width(scroll, panel, minimum=width)
        return scroll

    def _fit_controls_width(self, scroll: QScrollArea, panel: QWidget,
                            minimum: int = None):
        """
        Schedules a re-fit of a controls column to its content.

        Deferred to the next event-loop turn on purpose: when called straight
        after the input grids are rebuilt, Qt has not yet laid out the new
        widgets, so the size hints still describe the *previous* monomer count
        and the column ends up one step behind.
        """
        if scroll is None or panel is None:
            return
        QTimer.singleShot(0, lambda: self._apply_controls_width(
            scroll, panel, minimum))

    def _apply_controls_width(self, scroll: QScrollArea, panel: QWidget,
                              minimum: int = None):
        """
        Sizes a controls column to its content, so nothing needs horizontal
        scrolling and the plots keep whatever width is left over.

        Widens for extra monomers and narrows back when they are removed,
        clamped between a floor (keeps the 2-monomer layout from looking
        cramped) and a ceiling (stops the controls from crowding out the plots).
        """
        if scroll is None or panel is None:
            return
        try:
            scroll.isVisible()     # cheap probe: raises if C++ side is gone
        except RuntimeError:       # widget deleted before the timer fired
            return

        # Release the previous fixed width first: while it is applied the panel
        # is constrained by it, so its hints report the old width back and the
        # column could only ever grow.
        scroll.setMinimumWidth(0)
        scroll.setMaximumWidth(_QWIDGETSIZE_MAX)

        layout = panel.layout()
        if layout is not None:
            layout.activate()          # hints are stale until this runs

        # minimumSizeHint, not sizeHint: the minimum is the width below which
        # content is genuinely clipped, whereas the preferred width includes
        # slack for things like the log box that are happy to be compressed.
        floor = _CONTROLS_MIN_WIDTH if minimum is None else minimum
        # Reserve room for the vertical scrollbar; the column always has one.
        scrollbar = scroll.verticalScrollBar().sizeHint().width()
        needed = panel.minimumSizeHint().width() + scrollbar + _CONTROLS_PADDING
        scroll.setFixedWidth(max(floor, min(needed, _CONTROLS_MAX_WIDTH)))

    def create_simulator_tab(self):
        self.simulator_tab = QWidget()
        self.tabs.addTab(self.simulator_tab, "Simulator")
        sim_layout = QHBoxLayout(self.simulator_tab)

        controls_panel = QWidget()
        self.controls_layout = QVBoxLayout(controls_panel)
        self.controls_layout.setContentsMargins(8, 4, 8, 8)
        self.controls_layout.setSpacing(6)

        self.create_global_setup_group()
        self.create_rr_group()
        self.create_workflow_group()
        self.create_analysis_group()
        self.create_log_export_group()

        self.sim_progress = QProgressBar()
        self.sim_progress.setRange(0, 100)
        self.sim_progress.setValue(0)
        self.sim_progress.setTextVisible(False)
        self.controls_layout.addWidget(QLabel("Process Status:"))
        self.controls_layout.addWidget(self.sim_progress)
        self.controls_layout.addStretch()

        plots_panel = QWidget()
        plots_layout = QVBoxLayout(plots_panel)
        self.sim_fig = Figure(figsize=(12, 10))
        self.sim_canvas = FigureCanvas(self.sim_fig)
        plots_layout.addWidget(self.sim_canvas)

        self.sim_controls_panel = controls_panel
        self.sim_controls_scroll = self._wrap_in_scroll(controls_panel)
        sim_layout.addWidget(self.sim_controls_scroll)
        sim_layout.addWidget(plots_panel)
        # Plots absorb whatever width the controls column does not claim.
        sim_layout.setStretch(1, 1)

        self.update_rr_mr_inputs()
        self.reset_simulation()

    def create_analysis_tab(self):
        self.analysis_tab = QWidget()
        self.tabs.addTab(self.analysis_tab, "Analysis")
        analysis_layout = QHBoxLayout(self.analysis_tab)

        analysis_controls_panel = QWidget()
        analysis_controls_panel.setFixedWidth(450)
        self.analysis_controls_layout = QVBoxLayout(analysis_controls_panel)

        self.create_load_data_group()
        self.create_analysis_controls_group()

        self.analysis_plot_tabs = QTabWidget()
        self.setup_analysis_plots()

        analysis_layout.addWidget(analysis_controls_panel)
        analysis_layout.addWidget(self.analysis_plot_tabs)

    def setup_analysis_plots(self):
        self.analysis_tab_positional = QWidget()
        self.analysis_tab_distributions = QWidget()
        self.analysis_tab_sd = QWidget()
        self.analysis_plot_tabs.addTab(self.analysis_tab_positional, "Positional & Sequence")
        self.analysis_plot_tabs.addTab(self.analysis_tab_distributions, "Distribution Analysis")
        self.analysis_plot_tabs.addTab(self.analysis_tab_sd, "SD Metrics")

        layout1 = QVBoxLayout(self.analysis_tab_positional)
        self.analysis_fig1 = Figure(figsize=(10, 8))
        self.analysis_canvas1 = FigureCanvas(self.analysis_fig1)
        layout1.addWidget(self.analysis_canvas1)

        layout2 = QVBoxLayout(self.analysis_tab_distributions)
        self.analysis_fig2 = Figure(figsize=(14, 9))
        self.analysis_canvas2 = FigureCanvas(self.analysis_fig2)
        layout2.addWidget(self.analysis_canvas2)

        layout3 = QVBoxLayout(self.analysis_tab_sd)
        self.analysis_fig3 = Figure(figsize=(14, 8))
        self.analysis_canvas3 = FigureCanvas(self.analysis_fig3)
        layout3.addWidget(self.analysis_canvas3)

    def create_descriptor_space_tab(self):
        self.descriptor_space_tab = QWidget()
        self.tabs.addTab(self.descriptor_space_tab, "3D Descriptor Space")
        layout = QHBoxLayout(self.descriptor_space_tab)

        controls_panel = QWidget()
        controls_panel.setFixedWidth(450)
        controls_layout = QVBoxLayout(controls_panel)

        load_group = QGroupBox("1. Load & Analyze")
        load_layout = QVBoxLayout(load_group)

        group_layout = QHBoxLayout()
        group_layout.addWidget(QLabel("Assign to Group:"))
        self.cs_group_combo = QComboBox()
        self.cs_group_combo.setEditable(True)
        self.cs_group_combo.addItem("Group A")
        group_layout.addWidget(self.cs_group_combo)

        self.cs_load_btn = QPushButton("Load CSV(s) into Group")
        self.cs_load_btn.clicked.connect(self.load_csv_for_descriptor_space)
        self.cs_auto_add_check = QCheckBox("Add every Simulator block here automatically")
        self.cs_auto_add_check.setChecked(True)
        self.cs_auto_add_check.setToolTip(
            "Each finished block is added as 'Run N · block M' in the group "
            "'Simulator runs', with its full provenance.")
        # Editable table: double-click a name or a group to change it.
        self.cs_file_list = QTableWidget(0, 3)
        self.cs_file_list.setHorizontalHeaderLabels(["Name", "Group", "Chains"])
        self.cs_file_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.cs_file_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.cs_file_list.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.cs_file_list.verticalHeader().setVisible(False)
        self.cs_file_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cs_file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.cs_file_list.setEditTriggers(QAbstractItemView.DoubleClicked
                                          | QAbstractItemView.EditKeyPressed)
        self.cs_file_list.itemChanged.connect(self._on_cs_item_changed)
        cs_edit_row = QHBoxLayout()
        self.cs_move_btn = QPushButton("Move selected to group…")
        self.cs_move_btn.clicked.connect(self.cs_move_selected_to_group)
        self.cs_remove_btn = QPushButton("Remove selected")
        self.cs_remove_btn.clicked.connect(self.cs_remove_selected)
        cs_edit_row.addWidget(self.cs_move_btn)
        cs_edit_row.addWidget(self.cs_remove_btn)
        self.cs_clear_btn = QPushButton("Reset (Clear All Data)")
        self.cs_clear_btn.clicked.connect(self.clear_descriptor_space_list)

        self.cs_progress = QProgressBar()
        self.cs_progress.setValue(0)

        load_layout.addLayout(group_layout)
        load_layout.addWidget(self.cs_load_btn)
        load_layout.addWidget(self.cs_auto_add_check)
        load_layout.addWidget(QLabel("Loaded ensembles (double-click a name or group to edit):"))
        load_layout.addWidget(self.cs_file_list)
        load_layout.addLayout(cs_edit_row)
        load_layout.addWidget(self.cs_clear_btn)
        load_layout.addWidget(QLabel("Analysis Progress:"))
        load_layout.addWidget(self.cs_progress)

        results_group = QGroupBox("2. Analysis Results")
        results_layout = QVBoxLayout(results_group)
        self.cs_results_text = QTextEdit()
        self.cs_results_text.setReadOnly(True)
        results_layout.addWidget(self.cs_results_text)

        controls_layout.addWidget(load_group)
        controls_layout.addWidget(results_group)
        controls_layout.addStretch()

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.descriptor_space_fig = Figure(figsize=(10, 8))
        self.descriptor_space_canvas = FigureCanvas(self.descriptor_space_fig)
        self.descriptor_space_canvas.mpl_connect('pick_event', self.on_descriptor_space_pick)
        plot_layout.addWidget(self.descriptor_space_canvas)

        layout.addWidget(controls_panel)
        layout.addWidget(plot_panel)

    def create_batch_tab(self):
        self.batch_tab = QWidget()
        self.tabs.addTab(self.batch_tab, "Batch Simulator")
        batch_layout = QHBoxLayout(self.batch_tab)

        controls_panel = QWidget()
        self.batch_controls_layout = QVBoxLayout(controls_panel)
        self.batch_controls_layout.setContentsMargins(8, 4, 8, 8)
        self.batch_controls_layout.setSpacing(6)

        self.create_global_setup_group_batch()
        self.create_rr_group_batch()
        self.create_scan_setup_group_batch()
        self.create_workflow_group_batch()
        self.create_output_group_batch()
        self.batch_controls_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.batch_fig = Figure(figsize=(10, 6))
        self.batch_canvas = FigureCanvas(self.batch_fig)
        self.batch_ax = self.batch_fig.add_subplot(111)

        status_group = QGroupBox("Log & Controls")
        status_layout = QVBoxLayout(status_group)
        self.batch_log_text = QTextEdit()
        self.batch_log_text.setReadOnly(True)
        self.batch_log_text.append("Configure batch simulation and select an output directory.")
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)

        button_layout = QHBoxLayout()
        self.run_batch_btn = QPushButton("Run Batch Simulation")
        self.run_batch_btn.clicked.connect(self.run_batch_simulation)
        self.run_batch_btn.setFixedHeight(40)
        button_layout.addWidget(self.run_batch_btn)

        status_layout.addWidget(QLabel("Batch Log:"))
        status_layout.addWidget(self.batch_log_text)
        status_layout.addWidget(QLabel("Progress:"))
        status_layout.addWidget(self.batch_progress_bar)
        status_layout.addLayout(button_layout)

        right_layout.addWidget(self.batch_canvas)
        right_layout.addWidget(status_group)
        right_layout.setStretch(0, 3)
        right_layout.setStretch(1, 1)

        self.batch_controls_panel = controls_panel
        self.batch_controls_scroll = self._wrap_in_scroll(controls_panel)
        batch_layout.addWidget(self.batch_controls_scroll)
        batch_layout.addWidget(right_panel)
        batch_layout.setStretch(1, 1)
        self.update_rr_inputs_batch()

    # -----------------------------------------------------------------------
    # Control group builders (simulator tab)
    # -----------------------------------------------------------------------

    def create_global_setup_group(self):
        group_box = QGroupBox("1. Global Setup")
        layout = QGridLayout(group_box)

        self.n_mons_spin = QSpinBox()
        self.n_mons_spin.setRange(_MIN_MONOMERS, _MAX_MONOMERS)
        self.n_chains_spin = QSpinBox()
        self.n_chains_spin.setRange(_MIN_CHAINS, _MAX_CHAINS)
        self.n_chains_spin.setValue(_DEFAULT_CHAINS)

        # Optional seed (off by default — leaving it off lets the RNGs run free).
        self.seed_check = QCheckBox("Random seed:")
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(0)
        self.seed_spin.setEnabled(False)
        self.seed_check.toggled.connect(self.seed_spin.setEnabled)

        layout.addWidget(QLabel("Num Monomers:"), 0, 0)
        layout.addWidget(self.n_mons_spin, 0, 1)
        layout.addWidget(QLabel("Num Chains:"), 1, 0)
        layout.addWidget(self.n_chains_spin, 1, 1)
        layout.addWidget(self.seed_check, 2, 0)
        layout.addWidget(self.seed_spin, 2, 1)

        self.n_mons_spin.valueChanged.connect(self.update_rr_mr_inputs)
        self.controls_layout.addWidget(group_box)

    def create_rr_group(self):
        self.rr_group_box = QGroupBox("Reactivity Ratios (r_ij = k_ii / k_ij)")
        outer = QVBoxLayout(self.rr_group_box)
        self.rr_layout = QGridLayout()
        outer.addLayout(self.rr_layout)
        row = QHBoxLayout()
        self.rr_library_btn = QPushButton("From library…")
        self.rr_library_btn.setToolTip("Fill r_ij from the bundled reactivity-ratio library")
        self.rr_library_btn.clicked.connect(lambda: self.fill_r_from_library("sim"))
        self.rr_names_label = QLabel()
        self.rr_names_label.setWordWrap(True)
        row.addWidget(self.rr_library_btn)
        row.addWidget(self.rr_names_label, 1)
        outer.addLayout(row)
        self.controls_layout.addWidget(self.rr_group_box)

    def create_workflow_group(self):
        group_box = QGroupBox("2. Block-wise Workflow")
        layout = QVBoxLayout(group_box)

        self.mr_group_box = QGroupBox("Monomer Ratios (MRs)")
        self.mr_layout = QGridLayout(self.mr_group_box)
        layout.addWidget(self.mr_group_box)

        # PDI targeting
        self.target_pdi_check = QCheckBox("Target dispersity Đ (sets Ctr):")
        self.target_pdi_spin = QDoubleSpinBox()
        self.target_pdi_spin.setRange(1.0, 10.0)
        self.target_pdi_spin.setValue(_DEFAULT_PDI_TARGET)
        self.target_pdi_spin.setSingleStep(0.05)
        # 4 dp: at high DP the 1 + 1/DPn floor sits near 1.001, so 3 decimals
        # would round realistic targets onto the floor and read as unreachable.
        self.target_pdi_spin.setDecimals(4)
        self.target_pdi_spin.setEnabled(False)
        pdi_target_layout = QHBoxLayout()
        pdi_target_layout.addWidget(self.target_pdi_check)
        pdi_target_layout.addWidget(self.target_pdi_spin)
        layout.addLayout(pdi_target_layout)
        self.target_pdi_check.toggled.connect(self.toggle_pdi_targeting)

        # Simulation parameters
        param_layout = QGridLayout()
        self.block_len_spin = QSpinBox()
        self.block_len_spin.setRange(1, 10000)
        self.block_len_spin.setValue(_DEFAULT_BLOCK_LEN)

        self.conv_spin = QDoubleSpinBox()
        self.conv_spin.setRange(1.0, 100.0)
        self.conv_spin.setValue(_DEFAULT_CONVERSION)

        self.ctr_ideal_check = QCheckBox("Ideal exchange (Ctr → ∞)")
        self.ctr_ideal_check.setToolTip(
            "Instantaneous exchange: one monomer per activation, the narrowest "
            "(Poisson) limit.\nAn idealisation — no real RAFT agent achieves it."
        )
        self.ctr_ideal_check.toggled.connect(self._on_ctr_ideal_toggled)

        self.prune_spin = QSpinBox()
        self.prune_spin.setValue(0)

        param_layout.addWidget(QLabel("Monomer to CTA / DP at 100%"), 0, 0)
        param_layout.addWidget(self.block_len_spin, 0, 1)
        param_layout.addWidget(QLabel("Conversion (%):"), 1, 0)
        param_layout.addWidget(self.conv_spin, 1, 1)
        param_layout.addWidget(QLabel("Prune DP <:"), 2, 0)
        param_layout.addWidget(self.prune_spin, 2, 1)
        layout.addLayout(param_layout)

        # Ctr is indexed by terminal monomer, so it needs one input per monomer
        # type; the grid is rebuilt whenever the monomer count changes.
        # Title kept short: a group box cannot be narrower than its own title,
        # and this column is width-critical. Full meaning is in the tooltip.
        self.ctr_group_box = QGroupBox("Chain Transfer (Ctr)")
        self.ctr_group_box.setToolTip(
            "Chain transfer constant per terminal monomer — the unit carrying "
            "the radical\nsets both k_tr and k_p, so each monomer has its own "
            "value.\nLarger = faster degenerative exchange = narrower Đ.\n"
            "Indicative: <1 negligible control, ~10-100 moderate "
            "(trithiocarbonates),\n~100-1000 good (dithiobenzoates)."
        )
        self.ctr_layout = QGridLayout(self.ctr_group_box)
        layout.addWidget(self.ctr_group_box)
        layout.addWidget(self.ctr_ideal_check)

        self.retain_monomers_checkbox = QCheckBox("Retain residual monomers before next block")
        layout.addWidget(self.retain_monomers_checkbox)

        self.polymerize_btn = QPushButton("Polymerize First Block")
        self.polymerize_btn.clicked.connect(self.run_block)
        layout.addWidget(self.polymerize_btn)

        self.controls_layout.addWidget(group_box)

    def create_analysis_group(self):
        group_box = QGroupBox("4. Calculated Properties")
        layout = QGridLayout(group_box)

        self.pdi_display = QLineEdit("N/A")
        self.pdi_display.setReadOnly(True)
        self.lambda_display = QLineEdit("N/A")     # shows kappa* (neighbour correlation)
        self.lambda_display.setReadOnly(True)
        self.u_display = QLineEdit("N/A")
        self.u_display.setReadOnly(True)
        self.dp_display = QLineEdit("N/A")
        self.dp_display.setReadOnly(True)

        layout.addWidget(QLabel("Avg. DP:"), 0, 0)
        layout.addWidget(self.dp_display, 0, 1)
        layout.addWidget(QLabel("PDI (Đ):"), 1, 0)
        layout.addWidget(self.pdi_display, 1, 1)
        layout.addWidget(QLabel("κ* (alternating −1 … random 0 … blocky +1):"), 2, 0)
        layout.addWidget(self.lambda_display, 2, 1)
        layout.addWidget(QLabel("U (order: random 0 … fully predictable 1):"), 3, 0)
        layout.addWidget(self.u_display, 3, 1)

        self.controls_layout.addWidget(group_box)

    def create_log_export_group(self):
        group_box = QGroupBox("3. Log, Export & Reset")
        layout = QVBoxLayout(group_box)

        self.show_absolute_consumption_check = QCheckBox("Show Absolute Monomer Amounts")
        self.show_absolute_consumption_check.clicked.connect(
            self.trigger_consumption_plot_update
        )
        layout.addWidget(self.show_absolute_consumption_check)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        button_layout = QGridLayout()
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setToolTip("Ctrl+R")

        button_layout.addWidget(self.reset_btn, 0, 0)

        layout.addWidget(self.log_text)
        layout.addLayout(button_layout)
        self.controls_layout.addWidget(group_box)

        self.reset_btn.clicked.connect(self.reset_simulation)

    def create_load_data_group(self):
        group_box = QGroupBox("1. Load Data")
        layout = QVBoxLayout(group_box)
        self.load_csv_btn = QPushButton("Load Polymer CSV File")
        self.load_csv_btn.clicked.connect(self.load_csv_file)
        self.loaded_file_label = QLabel("No file loaded.")
        layout.addWidget(self.load_csv_btn)
        layout.addWidget(self.loaded_file_label)
        self.analysis_controls_layout.addWidget(group_box)

    def create_analysis_controls_group(self):
        group_box = QGroupBox("2. Actions")
        layout = QVBoxLayout(group_box)

        info_label = QLabel(
            "Metrics update automatically whenever you run a simulation "
            "or load a CSV file."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setValue(0)
        self.analysis_progress.setTextVisible(False)
        layout.addWidget(QLabel("Processing:"))
        layout.addWidget(self.analysis_progress)

        layout.addWidget(QLabel("Metrics:"))
        self.metrics_display = QTextEdit()
        self.metrics_display.setReadOnly(True)
        self.metrics_display.setMinimumHeight(200)
        self.metrics_display.setPlaceholderText(
            "Metrics appear automatically after a simulation or CSV load."
        )
        layout.addWidget(self.metrics_display)

        self.copy_metrics_btn = QPushButton("Copy Metrics to Clipboard")
        self.copy_metrics_btn.clicked.connect(self.copy_metrics_to_clipboard)
        self.copy_metrics_btn.setEnabled(False)
        layout.addWidget(self.copy_metrics_btn)

        self.analysis_controls_layout.addWidget(group_box)

    # -----------------------------------------------------------------------
    # Batch tab control group builders
    # -----------------------------------------------------------------------

    def create_global_setup_group_batch(self):
        group_box = QGroupBox("1. Global Setup")
        layout = QGridLayout(group_box)

        self.n_mons_spin_batch = QSpinBox()
        self.n_mons_spin_batch.setRange(_MIN_MONOMERS, _MAX_MONOMERS)
        self.n_chains_spin_batch = QSpinBox()
        self.n_chains_spin_batch.setRange(_MIN_CHAINS, _MAX_CHAINS)
        self.n_chains_spin_batch.setValue(_DEFAULT_CHAINS)

        # Optional base seed for parallel scans (each scan point gets seed + index).
        self.seed_check_batch = QCheckBox("Random seed:")
        self.seed_spin_batch = QSpinBox()
        self.seed_spin_batch.setRange(0, 2_147_483_647)
        self.seed_spin_batch.setValue(0)
        self.seed_spin_batch.setEnabled(False)
        self.seed_check_batch.toggled.connect(self.seed_spin_batch.setEnabled)

        layout.addWidget(QLabel("Num Monomers:"), 0, 0)
        layout.addWidget(self.n_mons_spin_batch, 0, 1)
        layout.addWidget(QLabel("Num Chains:"), 1, 0)
        layout.addWidget(self.n_chains_spin_batch, 1, 1)
        layout.addWidget(self.seed_check_batch, 2, 0)
        layout.addWidget(self.seed_spin_batch, 2, 1)

        self.n_mons_spin_batch.valueChanged.connect(self.update_rr_inputs_batch)
        self.batch_controls_layout.addWidget(group_box)

    def create_rr_group_batch(self):
        self.rr_group_box_batch = QGroupBox("Reactivity Ratios (r_ij = k_ii / k_ij)")
        outer = QVBoxLayout(self.rr_group_box_batch)
        self.rr_layout_batch = QGridLayout()
        outer.addLayout(self.rr_layout_batch)
        row = QHBoxLayout()
        self.rr_library_btn_batch = QPushButton("From library…")
        self.rr_library_btn_batch.setToolTip("Fill r_ij from the bundled reactivity-ratio library")
        self.rr_library_btn_batch.clicked.connect(lambda: self.fill_r_from_library("batch"))
        self.rr_names_label_batch = QLabel()
        self.rr_names_label_batch.setWordWrap(True)
        row.addWidget(self.rr_library_btn_batch)
        row.addWidget(self.rr_names_label_batch, 1)
        outer.addLayout(row)
        self.batch_controls_layout.addWidget(self.rr_group_box_batch)

    def create_scan_setup_group_batch(self):
        group_box = QGroupBox("2. Composition Scan")
        layout = QGridLayout(group_box)

        self.scan_mode_combo = QComboBox()
        self.scan_mode_combo.addItems(["Linear Scan (One Monomer)", "Custom Grid (N-1)"])
        self.scan_mode_combo.currentIndexChanged.connect(self.toggle_scan_mode_ui)
        layout.addWidget(QLabel("Scan Mode:"), 0, 0)
        layout.addWidget(self.scan_mode_combo, 0, 1)

        self.scan_block_combo = QComboBox()
        layout.addWidget(QLabel("Scan Target Block:"), 1, 0)
        layout.addWidget(self.scan_block_combo, 1, 1)

        self.scan_monomer_label = QLabel("Scan Monomer:")
        self.scan_monomer_combo = QComboBox()
        self.start_mr_label = QLabel("Start MR (%):")
        self.start_mr_spin = QDoubleSpinBox()
        self.start_mr_spin.setRange(0.0, 100.0)
        self.start_mr_spin.setValue(10.0)
        self.end_mr_label = QLabel("End MR (%):")
        self.end_mr_spin = QDoubleSpinBox()
        self.end_mr_spin.setRange(0.0, 100.0)
        self.end_mr_spin.setValue(50.0)
        self.num_steps_label = QLabel("Num Steps:")
        self.num_steps_spin = QSpinBox()
        self.num_steps_spin.setRange(2, _BATCH_SCAN_MAX_STEPS)
        self.num_steps_spin.setValue(5)

        layout.addWidget(self.scan_monomer_label, 2, 0)
        layout.addWidget(self.scan_monomer_combo, 2, 1)
        layout.addWidget(self.start_mr_label, 3, 0)
        layout.addWidget(self.start_mr_spin, 3, 1)
        layout.addWidget(self.end_mr_label, 4, 0)
        layout.addWidget(self.end_mr_spin, 4, 1)
        layout.addWidget(self.num_steps_label, 5, 0)
        layout.addWidget(self.num_steps_spin, 5, 1)

        self.grid_table = QTableWidget()
        self.grid_table.setColumnCount(3)
        self.grid_table.setHorizontalHeaderLabels(["Start %", "End %", "Step Size"])
        self.grid_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.grid_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.grid_table.setVisible(False)
        layout.addWidget(self.grid_table, 6, 0, 1, 2)

        self.batch_controls_layout.addWidget(group_box)

    def toggle_scan_mode_ui(self, index):
        is_grid = (index == 1)
        for widget in (self.scan_monomer_label, self.scan_monomer_combo,
                       self.start_mr_label, self.start_mr_spin,
                       self.end_mr_label, self.end_mr_spin,
                       self.num_steps_label, self.num_steps_spin):
            widget.setVisible(not is_grid)
        self.grid_table.setVisible(is_grid)
        if is_grid:
            self.setup_grid_table()

    def setup_grid_table(self):
        n_mons = self.n_mons_spin_batch.value()
        self.grid_table.setRowCount(n_mons)
        self.grid_table.setVerticalHeaderLabels(
            [f"Monomer {i + 1}" for i in range(n_mons)]
        )
        for i in range(n_mons):
            if i == n_mons - 1:
                item = QTableWidgetItem("Balance (100 - Sum)")
                item.setFlags(Qt.ItemIsEnabled)
                self.grid_table.setItem(i, 0, item)
                self.grid_table.setSpan(i, 0, 1, 3)
            else:
                for col, default in enumerate([0.0, 50.0, 10.0]):
                    sb = QDoubleSpinBox()
                    sb.setRange(0, 100)
                    sb.setValue(default)
                    self.grid_table.setCellWidget(i, col, sb)

    def create_workflow_group_batch(self):
        group_box = QGroupBox("3. Block Recipe")
        layout = QVBoxLayout(group_box)

        global_layout = QGridLayout()

        self.ctr_spin_batch = QDoubleSpinBox()
        self.ctr_spin_batch.setRange(0.01, _MAX_CTR)
        self.ctr_spin_batch.setValue(_DEFAULT_CTR)
        self.ctr_spin_batch.setDecimals(2)
        self.ctr_spin_batch.setSingleStep(1.0)

        self.prune_spin_batch = QSpinBox()
        self.prune_spin_batch.setValue(0)

        self.n_blocks_spin_batch = QSpinBox()
        self.n_blocks_spin_batch.setRange(1, 10)
        self.n_blocks_spin_batch.setValue(1)

        global_layout.addWidget(QLabel("Chain Transfer Const. (Ctr):"), 0, 0)
        global_layout.addWidget(self.ctr_spin_batch, 0, 1)
        global_layout.addWidget(QLabel("Prune DP <:"), 1, 0)
        global_layout.addWidget(self.prune_spin_batch, 1, 1)
        global_layout.addWidget(QLabel("Num Blocks:"), 2, 0)
        global_layout.addWidget(self.n_blocks_spin_batch, 2, 1)
        layout.addLayout(global_layout)

        self.block_recipe_table = QTableWidget()
        self.block_recipe_table.setFixedHeight(120)
        self.block_recipe_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.block_recipe_table.setSelectionMode(QAbstractItemView.NoSelection)
        layout.addWidget(self.block_recipe_table)

        recipe_btn_layout = QHBoxLayout()
        self.save_batch_recipe_btn = QPushButton("Save Recipe")
        self.load_batch_recipe_btn = QPushButton("Load Recipe")
        recipe_btn_layout.addWidget(self.save_batch_recipe_btn)
        recipe_btn_layout.addWidget(self.load_batch_recipe_btn)
        layout.addLayout(recipe_btn_layout)

        self.save_batch_recipe_btn.clicked.connect(self.save_batch_recipe)
        self.load_batch_recipe_btn.clicked.connect(self.load_batch_recipe)

        self.n_blocks_spin_batch.valueChanged.connect(self._rebuild_block_recipe_table)
        self.n_blocks_spin_batch.valueChanged.connect(self._update_scan_block_combo)

        self.batch_controls_layout.addWidget(group_box)

    def create_output_group_batch(self):
        group_box = QGroupBox("4. Output")
        layout = QVBoxLayout(group_box)

        self.output_dir_lineedit = QLineEdit("No directory selected")
        self.output_dir_lineedit.setReadOnly(True)
        self.select_dir_btn = QPushButton("Select Output Directory")
        self.select_dir_btn.clicked.connect(self.select_output_directory)

        cpu_count = os.cpu_count() or 1
        self.parallel_check = QCheckBox(
            f"Run scan points in parallel ({cpu_count} cores)"
        )
        self.parallel_check.setChecked(cpu_count > 1)
        self.parallel_check.setEnabled(cpu_count > 1)

        layout.addWidget(self.output_dir_lineedit)
        layout.addWidget(self.select_dir_btn)
        layout.addWidget(self.parallel_check)
        self.batch_controls_layout.addWidget(group_box)

    # -----------------------------------------------------------------------
    # Dynamic input helpers
    # -----------------------------------------------------------------------

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def update_rr_mr_inputs(self):
        n_mons = self.n_mons_spin.value()
        self.clear_layout(self.rr_layout)
        self.clear_layout(self.mr_layout)

        self.r_inputs = []
        for i in range(n_mons):
            row_inputs = []
            col_idx = 0
            for j in range(n_mons):
                if i == j:
                    continue
                spin = QDoubleSpinBox()
                spin.setDecimals(_R_DECIMALS)
                spin.setRange(0.0, 1000.0)
                spin.setValue(1.0)
                spin.valueChanged.connect(lambda _v: self._on_r_edited("sim"))
                label_text = f"r<sub>{i+1}</sub>:" if n_mons == 2 else f"r<sub>{i+1}{j+1}</sub>:"
                label = QLabel(label_text)
                self.rr_layout.addWidget(label, i, col_idx * 2)
                self.rr_layout.addWidget(spin, i, col_idx * 2 + 1)
                row_inputs.append(spin)
                col_idx += 1
            self.r_inputs.append(row_inputs)
        # A new monomer count invalidates any library naming of the slots.
        self._set_library_provenance("sim", None, None)

        self.mr_inputs = []
        for i in range(n_mons):
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(0.0, 100.0)
            spin.setValue(round(100.0 / n_mons, 2))
            spin.setMaximumWidth(_MONOMER_SPIN_WIDTH)
            label = QLabel(f"MR<sub>{i+1}</sub>:")
            row, col = divmod(i, _INPUTS_PER_ROW)
            self.mr_layout.addWidget(label, row, col * 2)
            self.mr_layout.addWidget(spin, row, col * 2 + 1)
            self.mr_inputs.append(spin)

        # Preserve existing Ctr values across a monomer-count change so that
        # editing the count does not silently reset the user's chemistry.
        previous = [s.value() for s in getattr(self, "ctr_inputs", [])]
        self.clear_layout(self.ctr_layout)
        self.ctr_inputs = []
        for i in range(n_mons):
            spin = QDoubleSpinBox()
            spin.setRange(0.01, _MAX_CTR)
            spin.setDecimals(2)
            spin.setSingleStep(1.0)
            spin.setValue(previous[i] if i < len(previous) else _DEFAULT_CTR)
            spin.setMaximumWidth(_MONOMER_SPIN_WIDTH)
            spin.setDisabled(self.ctr_ideal_check.isChecked()
                             or self.target_pdi_check.isChecked())
            label = QLabel(f"Ctr<sub>{i+1}</sub>:")
            # Wrap onto extra rows instead of one ever-widening row: vertical
            # growth is scrollable, horizontal growth is what crowds the plots.
            row, col = divmod(i, _INPUTS_PER_ROW)
            self.ctr_layout.addWidget(label, row, col * 2)
            self.ctr_layout.addWidget(spin, row, col * 2 + 1)
            self.ctr_inputs.append(spin)

        # The r-matrix, MR and Ctr grids all widened or narrowed; re-fit the
        # column so none of them needs horizontal scrolling.
        self._fit_controls_width(getattr(self, "sim_controls_scroll", None),
                                 getattr(self, "sim_controls_panel", None))

    def update_rr_inputs_batch(self):
        n_mons = self.n_mons_spin_batch.value()
        self.clear_layout(self.rr_layout_batch)

        self.r_inputs_batch = []
        for i in range(n_mons):
            row_inputs = []
            col_idx = 0
            for j in range(n_mons):
                if i == j:
                    continue
                spin = QDoubleSpinBox()
                spin.setDecimals(_R_DECIMALS)
                spin.setRange(0.0, 1000.0)
                spin.setValue(1.0)
                spin.valueChanged.connect(lambda _v: self._on_r_edited("batch"))
                label_text = f"r<sub>{i+1}</sub>:" if n_mons == 2 else f"r<sub>{i+1}{j+1}</sub>:"
                self.rr_layout_batch.addWidget(QLabel(label_text), i, col_idx * 2)
                self.rr_layout_batch.addWidget(spin, i, col_idx * 2 + 1)
                row_inputs.append(spin)
                col_idx += 1
            self.r_inputs_batch.append(row_inputs)
        self._set_library_provenance("batch", None, None)

        self.scan_monomer_combo.clear()
        self.scan_monomer_combo.addItems([f"Monomer {i + 1}" for i in range(n_mons)])

        if self.scan_mode_combo.currentIndex() == 1:
            self.setup_grid_table()

        self._rebuild_block_recipe_table()
        self._update_scan_block_combo()

        # The r-matrix and block-recipe table both widened; re-fit the column.
        self._fit_controls_width(getattr(self, "batch_controls_scroll", None),
                                 getattr(self, "batch_controls_panel", None))

    def _rebuild_block_recipe_table(self):
        n_mons = self.n_mons_spin_batch.value()
        n_blocks = self.n_blocks_spin_batch.value()
        col_headers = [f"M{i + 1} %" for i in range(n_mons)] + ["DP", "Conv %", "Purge"]
        self.block_recipe_table.setColumnCount(n_mons + 3)
        self.block_recipe_table.setHorizontalHeaderLabels(col_headers)
        self.block_recipe_table.setRowCount(n_blocks)
        self.block_recipe_table.setVerticalHeaderLabels(
            [f"Block {b + 1}" for b in range(n_blocks)]
        )
        default_mr = round(100.0 / n_mons, 1)
        for b in range(n_blocks):
            for m in range(n_mons):
                sb = QDoubleSpinBox()
                sb.setRange(0.0, 100.0)
                sb.setDecimals(1)
                sb.setValue(default_mr)
                self.block_recipe_table.setCellWidget(b, m, sb)
            dp_sb = QSpinBox()
            dp_sb.setRange(1, 10000)
            dp_sb.setValue(_DEFAULT_BLOCK_LEN)
            self.block_recipe_table.setCellWidget(b, n_mons, dp_sb)
            conv_sb = QDoubleSpinBox()
            conv_sb.setRange(1.0, 100.0)
            conv_sb.setDecimals(1)
            conv_sb.setValue(_DEFAULT_CONVERSION)
            self.block_recipe_table.setCellWidget(b, n_mons + 1, conv_sb)
            purge_cb = QCheckBox()
            purge_cb.setChecked(b > 0)
            purge_cb.setEnabled(b > 0)
            purge_container = QWidget()
            purge_layout = QHBoxLayout(purge_container)
            purge_layout.addWidget(purge_cb)
            purge_layout.setAlignment(Qt.AlignCenter)
            purge_layout.setContentsMargins(0, 0, 0, 0)
            self.block_recipe_table.setCellWidget(b, n_mons + 2, purge_container)

    def _update_scan_block_combo(self):
        n_blocks = self.n_blocks_spin_batch.value()
        current_idx = self.scan_block_combo.currentIndex()
        self.scan_block_combo.clear()
        self.scan_block_combo.addItems([f"Block {b + 1}" for b in range(n_blocks)])
        if 0 <= current_idx < n_blocks:
            self.scan_block_combo.setCurrentIndex(current_idx)

    def _read_block_recipe(self) -> list:
        n_mons = self.n_mons_spin_batch.value()
        n_blocks = self.n_blocks_spin_batch.value()
        recipe = []
        for b in range(n_blocks):
            mrs = []
            for m in range(n_mons):
                widget = self.block_recipe_table.cellWidget(b, m)
                mrs.append(widget.value() if widget else round(100.0 / n_mons, 1))
            dp_widget = self.block_recipe_table.cellWidget(b, n_mons)
            conv_widget = self.block_recipe_table.cellWidget(b, n_mons + 1)
            purge_container = self.block_recipe_table.cellWidget(b, n_mons + 2)
            dp = dp_widget.value() if dp_widget else _DEFAULT_BLOCK_LEN
            conv = conv_widget.value() if conv_widget else _DEFAULT_CONVERSION
            purge = False
            if purge_container:
                cb = purge_container.findChild(QCheckBox)
                if cb:
                    purge = cb.isChecked()
            recipe.append({'mrs': mrs, 'dp': dp, 'conv': conv, 'purge': purge})
        return recipe

    # -----------------------------------------------------------------------
    # Reactivity-ratio library
    # -----------------------------------------------------------------------

    def _lib_state(self, tab: str) -> dict:
        attr = "_lib_state_sim" if tab == "sim" else "_lib_state_batch"
        if not hasattr(self, attr):
            setattr(self, attr, {"names": None, "sources": None, "edited": False,
                                 "filling": False})
        return getattr(self, attr)

    def _set_library_provenance(self, tab: str, names, sources):
        st = self._lib_state(tab)
        st.update(names=list(names) if names else None,
                  sources=dict(sources) if sources else None, edited=False)
        self._update_names_label(tab)

    def _update_names_label(self, tab: str):
        label = getattr(self, "rr_names_label" if tab == "sim" else "rr_names_label_batch", None)
        if label is None:
            return
        st = self._lib_state(tab)
        if not st["names"]:
            label.setText("<i>Monomers not named</i>")
            return
        text = " / ".join(reactivity_library.abbreviation(n) or n for n in st["names"])
        if st["sources"]:
            text += f" — {len(st['sources'])} pair(s) from library"
        if st["edited"]:
            text += " (edited by hand since)"
        label.setText(text)

    def _on_r_edited(self, tab: str):
        st = self._lib_state(tab)
        if not st["filling"] and st["sources"] and not st["edited"]:
            st["edited"] = True
            self._update_names_label(tab)

    def fill_r_from_library(self, tab: str):
        """Opens the library picker and writes the chosen values into r_ij."""
        if not reactivity_library.available():
            QMessageBox.warning(self, "Library not found",
                                "The reactivity-ratio data file is missing.")
            return
        from .library_dialogs import FillFromLibraryDialog, apply_selection_to_inputs
        n = (self.n_mons_spin if tab == "sim" else self.n_mons_spin_batch).value()
        r_inputs = self.r_inputs if tab == "sim" else self.r_inputs_batch
        st = self._lib_state(tab)
        dlg = FillFromLibraryDialog(n, current_names=st["names"], parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
        st["filling"] = True
        try:
            sources = apply_selection_to_inputs(r_inputs, dlg.selection)
        finally:
            st["filling"] = False
        self._set_library_provenance(tab, dlg.monomer_names, sources)
        missing = [k for k, v in dlg.selection.items() if v is None]
        log = self.log if tab == "sim" else self.log_batch
        log("Reactivity ratios filled from library: " + ", ".join(
            f"M{k}={v}" for k, v in sources.items()) if sources else
            "No library values applied.")
        if missing:
            log("No library data for pair(s) " + ", ".join(
                f"M{i + 1}–M{j + 1}" for i, j in missing) + "; their r values were kept.")

    def _warn_r_matrix(self, r_values, log) -> bool:
        """Shows r-matrix warnings; returns False if the user cancels."""
        warnings = simulation_engine.check_r_values(r_values)
        if not warnings:
            return True
        for w in warnings:
            log(f"WARNING: {w}")
        reply = QMessageBox.warning(
            self, "Check reactivity ratios", "\n\n".join(warnings) + "\n\nRun anyway?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        return reply == QMessageBox.Yes

    # -----------------------------------------------------------------------
    # Logging helpers
    # -----------------------------------------------------------------------

    def log(self, message: str):
        self.log_text.append(message)
        logger.info(message)

    def log_batch(self, message: str):
        self.batch_log_text.append(message)
        logger.info("[BATCH] %s", message)

    # -----------------------------------------------------------------------
    # Simulation control
    # -----------------------------------------------------------------------

    def reset_simulation(self):
        """Resets the simulator tab to its initial state, clearing the engine and all plots."""
        self.engine = None
        self._run_setup = {}
        self.is_initialized = False
        self.log_text.clear()
        self.log("Welcome. Configure setup.")
        if app_icon_path() is None:
            self.log("Note: the program icon file was not found in this build.")
        self.polymerize_btn.setText("Polymerize First Block")
        self.polymerize_btn.setEnabled(True)
        self.pdi_display.setText("N/A")
        self.lambda_display.setText("N/A")
        self.u_display.setText("N/A")
        self.dp_display.setText("N/A")
        self.retain_monomers_checkbox.setVisible(False)
        self.retain_monomers_checkbox.setChecked(False)
        self.show_absolute_consumption_check.setChecked(False)
        self._set_ctr_controls_locked(False)
        self.target_pdi_check.setChecked(True)        # back to the default mode
        self.last_conv_hist = None
        self.last_amounts_hist = None
        self.sim_fig.clear()
        self.setup_plot_layout()
        self.ax_consumption.set_title("Monomer Consumption")
        self.ax_composition.set_title("Normalized Monomer Occurrence")
        self.ax_pos_density.set_title("Monomer Positional Density")
        self.ax_heatmap.set_title("Sequence Visualization")
        self.ax_kde_top.set_title("Binary cone of heterogeneity")
        self.sim_fig.subplots_adjust(left=0.07, right=0.94, top=0.95, bottom=0.08)
        self.sim_canvas.draw()
        self.sim_progress.setRange(0, 100)
        self.sim_progress.setValue(0)

    def setup_plot_layout(self):
        """Creates the 6×2 subplot grid for the simulator tab plots."""
        gs = self.sim_fig.add_gridspec(6, 2, wspace=0.3, hspace=1.6)
        self.ax_consumption = self.sim_fig.add_subplot(gs[0:3, 0])
        self.ax_composition = self.sim_fig.add_subplot(gs[3:6, 0])
        self.ax_pos_density = self.sim_fig.add_subplot(gs[0:2, 1])
        self.ax_heatmap = self.sim_fig.add_subplot(gs[2:4, 1])
        inner_gs = gs[4:6, 1].subgridspec(
            2, 2, wspace=0.05, hspace=0.05,
            width_ratios=[3, 1], height_ratios=[1, 3],
        )
        self.ax_kde_main = self.sim_fig.add_subplot(inner_gs[1, 0])
        self.ax_kde_top = self.sim_fig.add_subplot(inner_gs[0, 0], sharex=self.ax_kde_main)
        self.ax_kde_right = self.sim_fig.add_subplot(inner_gs[1, 1], sharey=self.ax_kde_main)

    def run_block(self):
        """Start polymerisation of the next block on a background thread."""
        try:
            self.sim_progress.setRange(0, 0)
            self.sim_progress.setTextVisible(False)
            self.polymerize_btn.setEnabled(False)

            initial_pdi_data = None
            if self.is_initialized:
                initial_pdi_data = analysis.calculate_pdi(
                    [p.asArray() for p in self.engine.polymerArray]
                )

            # Ctr is a property of the RAFT agent, so it is fixed for the whole
            # synthesis: only block 1 resolves it, later blocks inherit it.
            if not self.is_initialized:
                if self.target_pdi_check.isChecked():
                    solved = self.find_ctr_for_target_pdi()
                    if solved is None:
                        self._restore_sim_ui()
                        return
                    # Broadcast the solved uniform value across all monomers,
                    # then read back so ctr is always a per-monomer list.
                    self._set_ctr_widgets(solved)
                ctr = self._current_ctr()
            else:
                ctr = self.engine.Ctr

            mrs = [s.value() for s in self.mr_inputs]
            block_len = self.block_len_spin.value()
            conversion = self.conv_spin.value()

            if not self.is_initialized:
                params = {
                    'N_MONs': self.n_mons_spin.value(),
                    'N_CHAINs': self.n_chains_spin.value(),
                    'R_values': [[s.value() for s in row] for row in self.r_inputs],
                    'Ctr': ctr,
                }
                if not self._warn_r_matrix(params['R_values'], self.log):
                    self._restore_sim_ui()
                    return
                # Snapshot of the synthesis-level setup actually used; exports
                # and recipes read this, not the (later editable) widgets.
                st = self._lib_state("sim")
                self._run_setup = {
                    "monomer_names": st["names"],
                    "r_sources": (st["sources"] if st["sources"] and not st["edited"]
                                  else None),
                }
                if self.seed_check.isChecked():
                    params['seed'] = self.seed_spin.value()
                    self.log(f"Seeding RNGs with {params['seed']} for reproducibility.")
                self.engine = simulation_engine.SimulationEngine(params)
                self.log("Engine initialized. Polymerizing block 1...")

                self._sim_worker = _SimWorker(
                    self.engine, mrs, block_len, conversion,
                    is_first_block=True,
                )
            else:
                self.log(f"Polymerizing Block {self.engine.block_num + 1}...")
                self.log(f"--> Inherited Ctr = {self._fmt_ctr(ctr)} "
                         f"(same RAFT agent); this block's Đ is a prediction.")
                purge = not self.retain_monomers_checkbox.isChecked()
                self.log("--> Residual monomers purged." if purge
                         else "--> Residual monomers retained.")
                self._sim_worker = _SimWorker(
                    self.engine, mrs, block_len, conversion,
                    is_first_block=False,
                    purge_monomers=purge,
                )

            self._initial_pdi_data_for_block = initial_pdi_data
            self._sim_worker.finished.connect(self._on_block_finished)
            self._sim_worker.error.connect(self._on_sim_error)
            self._sim_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
            self.log(f"ERROR: {e}")
            self._restore_sim_ui()

    def _on_block_finished(self, conv_hist, amounts_hist, status):
        """
        Slot called by ``_SimWorker.finished`` when a propagation block completes.

        Updates UI state, logs completion, warns on "stuck" status, and
        triggers metrics / plot refresh.
        """
        try:
            try:
                self._cs_add_simulation_run()
            except Exception as exc:                     # never block the simulator
                self.log(f"Could not add run to the 3D Descriptor Space: {exc}")
            if not self.is_initialized:
                self.is_initialized = True
                self.polymerize_btn.setText("Polymerize Next Block")
                self.retain_monomers_checkbox.setVisible(True)
                # The RAFT agent is chosen once; lock it for later blocks.
                self._set_ctr_controls_locked(True)
                self.log("Block 1 complete.")
            else:
                self.log(f"Block {self.engine.block_num} complete.")

            if status == "stuck":
                self.log("\nWARNING: Polymerization stopped early.")
                self.log("No available monomers could be added to any chain.")
                QMessageBox.warning(self, "Simulation Stuck",
                                    "Polymerization stopped early — no chain could "
                                    "accept any available monomer.")
                self.polymerize_btn.setEnabled(False)

            self.last_conv_hist = conv_hist
            self.last_amounts_hist = amounts_hist
            self.update_analytics_and_plots(conv_hist, amounts_hist)

            final_comp = analysis.calculate_average_composition(
                [p.asArray() for p in self.engine.polymerArray]
            )
            comp_str = ", ".join([f"M{k}: {v:.1f}%" for k, v in final_comp.items()])
            self.log(f"Final Average Polymer Composition: {comp_str}")

            if (self.is_initialized
                    and self._initial_pdi_data_for_block
                    and self.target_pdi_check.isChecked()):
                try:
                    final_pdi = analysis.calculate_pdi(
                        [p.asArray() for p in self.engine.polymerArray]
                    )
                    mu1 = self._initial_pdi_data_for_block['Mn_DP']
                    D1 = self._initial_pdi_data_for_block['PDI']
                    mu_final = final_pdi['Mn_DP']
                    D_final = final_pdi['PDI']
                    mu2 = mu_final - mu1
                    if mu2 > 0:
                        D_added = (
                            1 + ((mu_final ** 2 * (D_final - 1)) - (mu1 ** 2 * (D1 - 1)))
                            / mu2 ** 2
                        )
                        self.log(f"Apparent dispersity (Đ) of added block: {D_added:.3f}")
                except Exception as e:
                    self.log(f"Could not calculate apparent dispersity: {e}")

            # Keep the Analysis tab in sync with every completed block: transfer
            # the ensemble and recompute the full metric set (including SD)
            # automatically, without stealing focus from the Simulator tab.
            if self._auto_update_analysis():
                self.log("Analysis tab updated with the latest ensemble.")
        finally:
            self._restore_sim_ui()

    def _on_sim_error(self, message: str):
        """Slot called by ``_SimWorker.error``; shows a critical dialog and restores the UI."""
        QMessageBox.critical(self, "Simulation Error", message)
        self.log(f"ERROR: {message}")
        self._restore_sim_ui()

    def _restore_sim_ui(self):
        """Re-enables the Polymerize button and resets the progress bar after a block run."""
        self.sim_progress.setRange(0, 100)
        self.sim_progress.setValue(100)
        self.polymerize_btn.setEnabled(True)

    # -----------------------------------------------------------------------
    # Analytics and plots
    # -----------------------------------------------------------------------

    def update_analytics_and_plots(self, conv_hist, amounts_hist):
        """
        Recalculates PDI / eigenvalue metrics and redraws all simulator-tab plots.

        Called after each block completes.  Reads the current polymer ensemble
        from ``self.engine`` and refreshes all five plot panels.
        """
        self.sim_fig.clear()
        self.setup_plot_layout()

        polymer_arrays = [p.asArray() for p in self.engine.polymerArray]
        pdi_data = analysis.calculate_pdi(polymer_arrays)
        kappa = analysis.calculate_kappa_metrics(polymer_arrays)

        self.dp_display.setText(f"{pdi_data['Mn_DP']:.2f}")
        self.pdi_display.setText(f"{pdi_data['PDI']:.3f}")
        self.lambda_display.setText(f"{kappa['kappa_star']:+.4f}")
        self.u_display.setText(f"{analysis.calculate_uncertainty_coefficient(polymer_arrays):.4f}")

        show_absolute = self.show_absolute_consumption_check.isChecked()
        plotting.plot_monomer_consumption(
            self.ax_consumption, conv_hist, amounts_hist,
            self.engine.N_MONs, normalize=not show_absolute,
        )
        plotting.plot_mirrored_kde(self.ax_pos_density, polymer_arrays, self.engine.N_MONs)
        plotting.plot_sequence_heatmap(self.ax_heatmap, polymer_arrays, 40, self.engine.N_MONs)

        occurrence_history = analysis.calculate_monomer_occurrence(
            self.engine.initial_monomer_amounts_in_block, amounts_hist
        )
        plotting.plot_monomer_occurrence(
            self.ax_composition, conv_hist, occurrence_history, self.engine.N_MONs
        )
        plotting.plot_2d_kde_with_marginals(
            (self.ax_kde_main, self.ax_kde_top, self.ax_kde_right),
            polymer_arrays, self.engine.N_MONs,
        )
        self.sim_fig.subplots_adjust(left=0.07, right=0.94, top=0.95, bottom=0.08)
        self.sim_canvas.draw()

    def trigger_consumption_plot_update(self):
        """Re-renders the consumption plot when the normalise/absolute toggle changes."""
        if self.last_conv_hist is not None and self.last_amounts_hist is not None:
            show_absolute = self.show_absolute_consumption_check.isChecked()
            plotting.plot_monomer_consumption(
                self.ax_consumption, self.last_conv_hist, self.last_amounts_hist,
                self.engine.N_MONs, normalize=not show_absolute,
            )
            self.sim_canvas.draw()

    # -----------------------------------------------------------------------
    # Ctr helpers and analytic PDI targeting
    # -----------------------------------------------------------------------

    def _on_ctr_ideal_toggled(self, checked: bool):
        for spin in self.ctr_inputs:
            spin.setDisabled(checked or self.target_pdi_check.isChecked())

    def _current_ctr(self):
        """
        Chain transfer constants currently selected, one per terminal monomer.

        Returns a list matching the monomer count, which is what the engine
        expects; a uniform vector is equivalent to the old scalar behaviour.
        """
        if self.ctr_ideal_check.isChecked():
            return [simulation_engine.IDEAL_EXCHANGE] * len(self.ctr_inputs)
        return [float(s.value()) for s in self.ctr_inputs]

    def _set_ctr_widgets(self, ctr):
        """
        Reflects a Ctr value back into the per-monomer spinboxes.

        Accepts a scalar (applied to every monomer) or a per-monomer sequence.
        """
        values = ctr if isinstance(ctr, (list, tuple)) else [ctr] * len(self.ctr_inputs)
        if all(v is None or not math.isfinite(float(v)) for v in values):
            self.ctr_ideal_check.setChecked(True)
            return
        self.ctr_ideal_check.setChecked(False)
        for spin, val in zip(self.ctr_inputs, values):
            spin.setValue(min(float(val), _MAX_CTR))

    @staticmethod
    def _fmt_ctr(ctr) -> str:
        """Renders a scalar or per-monomer Ctr for the log."""
        if isinstance(ctr, (list, tuple)):
            if len({float(c) for c in ctr}) == 1:
                return CompositionalSuiteApp._fmt_ctr(ctr[0])
            return "[" + ", ".join(CompositionalSuiteApp._fmt_ctr(c) for c in ctr) + "]"
        return "∞ (ideal)" if not math.isfinite(float(ctr)) else f"{float(ctr):.3g}"

    def find_ctr_for_target_pdi(self):
        """
        Solves the target dispersity for the required chain transfer constant.

        Inverts Đ = 1 + 1/DPn + (2-x)/(x*Ctr) analytically — no simulation
        sweep — and reports back on whether the answer corresponds to
        chemistry a real RAFT agent could deliver.

        Returns:
            The required Ctr, or None if the target is unreachable or the user
            declines an implausible one.
        """
        target_pdi = float(self.target_pdi_spin.value())
        block_len = float(self.block_len_spin.value())
        conversion = float(self.conv_spin.value())

        dpn = simulation_engine.dpn_at_conversion(block_len, conversion)
        floor = simulation_engine.dispersity_floor(block_len, conversion)

        self.log(f"\n--- Solving Ctr for target Đ = {target_pdi:.3f} ---")
        self.log(f"DPn at {conversion:.0f}% conversion = {dpn:.1f}, "
                 f"narrowest achievable Đ = {floor:.4f}")

        ctr = simulation_engine.ctr_for_target_dispersity(
            target_pdi, block_len, conversion
        )

        # --- Infeasible: below the 1 + 1/DPn floor, at any Ctr whatsoever.
        if ctr is None:
            self.log(f"INFEASIBLE: Đ = {target_pdi:.3f} is at or below the floor.")
            QMessageBox.warning(
                self, "Target dispersity unreachable",
                f"Đ = {target_pdi:.3f} cannot be reached at these conditions.\n\n"
                f"At DPn = {dpn:.1f} the narrowest possible dispersity is "
                f"Đ = {floor:.4f} (the Poisson limit, reached only as "
                f"Ctr → ∞). No chain transfer constant, however large, goes "
                f"below it.\n\n"
                f"To narrow the distribution further, raise the target DP "
                f"(currently {block_len:.0f}) or the conversion "
                f"(currently {conversion:.0f}%).",
            )
            return None

        label, note, plausible = simulation_engine.classify_ctr(ctr)
        self.log(f"Required Ctr = {ctr:.4g}  [{label}]")

        # --- Feasible, but outside reported RAFT chemistry: warn, let them proceed.
        if not plausible:
            self.log(f"WARNING: Ctr = {ctr:.4g} is {label}.")
            reply = QMessageBox.question(
                self, "Implausible chain transfer constant",
                f"Đ = {target_pdi:.3f} at DPn = {dpn:.1f} requires\n"
                f"Ctr = {ctr:.4g}  —  {label}.\n\n"
                f"{note}\n\n"
                f"Run the simulation with this value anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                self.log("--- Cancelled at user request. ---")
                return None
        else:
            self.log(f"Plausible regime: {note}")

        self.log(f"--- Using Ctr = {ctr:.4g} ---")
        return ctr

    # -----------------------------------------------------------------------
    # Export / save
    # -----------------------------------------------------------------------

    def export_data(self):
        """Saves the current polymer ensemble as a CSV file (one chain per row)."""
        if not self.engine or not self.engine.polymerArray:
            QMessageBox.warning(self, "Warning", "No data to export.")
            return
        prune_dp = self.prune_spin.value()
        polymers_to_export = [
            p.asArray() for p in self.engine.polymerArray if p.len() >= prune_dp
        ]
        if not polymers_to_export:
            QMessageBox.warning(self, "Warning", f"No polymers with DP >= {prune_dp}.")
            return
        mrs_str = str([round(s.value(), 2) for s in self.mr_inputs])
        fname = (f"NM{self.engine.N_MONs}_NC{self.engine.N_CHAINs}"
                 f"_B{self.engine.block_num}_MR{mrs_str}"
                 f"_Ctr{recipe_io.ctr_tag(self.engine.Ctr)}_FILT{prune_dp}.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Save Sequences", fname, "CSV Files (*.csv)")
        if path:
            try:
                setup = self._engine_global_setup()
                setup["prune_dp"] = prune_dp
                meta = recipe_io.build_dataset_metadata(
                    setup, [dict(b) for b in self.engine.block_history],
                    extra={"source": "Compositional Suite Simulator",
                           "n_blocks_completed": self.engine.block_num},
                )
                recipe_io.write_chains_csv(path, polymers_to_export, metadata=meta)
                self.log(f"Exported {len(polymers_to_export)} sequences "
                         f"(+ metadata sidecar {os.path.basename(recipe_io.sidecar_path(path))}).")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed: {e}")
                self.log(f"ERROR: {e}")

    def export_plots(self):
        if not self.engine:
            QMessageBox.warning(self, "No Data", "Run simulation first.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not directory:
            return
        try:
            self.save_ax_as_svg(self.ax_consumption,
                                os.path.join(directory, "1_consumption.svg"))
            self.save_kde_grid_as_svg(os.path.join(directory, "2_count_dist.svg"))
            self.save_ax_as_svg(self.ax_pos_density,
                                os.path.join(directory, "3_positional_density.svg"))
            self.save_ax_as_svg(self.ax_heatmap,
                                os.path.join(directory, "4_sequence_viz.svg"))
            self.save_ax_as_svg(self.ax_composition,
                                os.path.join(directory, "5_composition.svg"))
            self.log(f"Exported 5 plots to:\n{directory}")
            QMessageBox.information(self, "Success", "Plots exported.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")
            self.log(f"ERROR: {e}")

    def export_analysis_plots(self):
        if not self.loaded_polymers:
            QMessageBox.warning(self, "No Data", "Load a CSV file before exporting.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save Plots")
        if not directory:
            return
        try:
            path1 = os.path.join(directory, "analysis_positional_sequence.svg")
            path2 = os.path.join(directory, "analysis_distributions.svg")
            self.analysis_fig1.savefig(path1, format='svg', bbox_inches='tight',
                                       metadata=self._svg_metadata("analysis"))
            self.analysis_fig2.savefig(path2, format='svg', bbox_inches='tight',
                                       metadata=self._svg_metadata("analysis"))
            self.log(f"Exported 2 analysis plot sheets to:\n{directory}")
            QMessageBox.information(self, "Success", "Analysis plots exported successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export plots:\n{e}")
            self.log(f"ERROR exporting analysis plots: {e}")

    def export_analysis_csv(self):
        if not self.loaded_polymers:
            QMessageBox.warning(self, "No Data", "Load a CSV file before exporting.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save CSVs")
        if not directory:
            return
        try:
            import csv as _csv
            all_mons = [m for p in self.loaded_polymers for m in p]
            n_mons = max(all_mons) if all_mons else 1
            files_written = []

            # 1. Sequences (raw chains)
            p = os.path.join(directory, "analysis_sequences.csv")
            with open(p, 'w', newline='') as f:
                _csv.writer(f).writerows(self.loaded_polymers)
            files_written.append("analysis_sequences.csv")

            # 2. Composition per chain
            p = os.path.join(directory, "analysis_composition.csv")
            with open(p, 'w', newline='') as f:
                w = _csv.writer(f)
                w.writerow(['chain_index', 'dp'] +
                            [f'f_M{i + 1}' for i in range(n_mons)])
                for idx, chain in enumerate(self.loaded_polymers):
                    L = len(chain)
                    if L == 0:
                        continue
                    from collections import Counter
                    c = Counter(chain)
                    w.writerow([idx, L] + [c.get(i + 1, 0) / L for i in range(n_mons)])
            files_written.append("analysis_composition.csv")

            # 3. DP distribution
            p = os.path.join(directory, "analysis_dp_distribution.csv")
            dps = [len(chain) for chain in self.loaded_polymers if chain]
            with open(p, 'w', newline='') as f:
                w = _csv.writer(f)
                w.writerow(['dp'])
                for dp in dps:
                    w.writerow([dp])
            files_written.append("analysis_dp_distribution.csv")

            # 4. Positional histogram (smoothed positional distribution data)
            hist_data, mon_fr = analysis.get_normalized_positional_histogram_data(
                self.loaded_polymers, 60, True
            )
            if hist_data is not None and not hist_data.empty:
                p = os.path.join(directory, "analysis_positional_histogram.csv")
                hist_data.to_csv(p)
                files_written.append("analysis_positional_histogram.csv")

            # 5. Run-length counts per monomer
            p = os.path.join(directory, "analysis_run_lengths.csv")
            with open(p, 'w', newline='') as f:
                w = _csv.writer(f)
                w.writerow(['monomer', 'run_length', 'count'])
                from collections import Counter as _Counter
                for mid in range(1, n_mons + 1):
                    runs = []
                    for chain in self.loaded_polymers:
                        if not chain:
                            continue
                        cur, cnt = chain[0], 1
                        for m in chain[1:]:
                            if m == cur:
                                cnt += 1
                            else:
                                if cur == mid:
                                    runs.append(cnt)
                                cur, cnt = m, 1
                        if cur == mid:
                            runs.append(cnt)
                    for length, count in sorted(_Counter(runs).items()):
                        w.writerow([f'M{mid}', length, count])
            files_written.append("analysis_run_lengths.csv")

            # 6. Monomer occurrence vs conversion (if history data exists)
            comp_data = analysis.get_composition_data(self.loaded_polymers, n_mons)
            if comp_data is not None and not comp_data.empty:
                p = os.path.join(directory, "analysis_composition_kde_data.csv")
                comp_data.to_csv(p, index=False)
                files_written.append("analysis_composition_kde_data.csv")

            self.log(f"Exported {len(files_written)} CSV files to:\n{directory}")
            QMessageBox.information(
                self, "Success",
                f"Exported {len(files_written)} CSV files:\n" +
                "\n".join(files_written)
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSVs:\n{e}")
            self.log(f"ERROR exporting analysis CSVs: {e}")

    def export_descriptor_space_plots(self):
        if not self.descriptor_space_data:
            QMessageBox.warning(self, "No Data", "No descriptor-space data to export.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not directory:
            return
        try:
            filepath = os.path.join(directory, "Descriptor_Space.svg")
            self.descriptor_space_fig.savefig(filepath, format='svg', bbox_inches='tight',
                                        metadata=self._svg_metadata("descriptor_space"))
            self.log(f"Exported Descriptor-space plots to:\n{directory}")
            QMessageBox.information(self, "Success", "Descriptor-space plots exported successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")
            self.log(f"ERROR: {e}")

    def export_descriptor_space_csv(self):
        """Writes the descriptor-space coordinates of every loaded ensemble to a CSV file."""
        if not self.descriptor_space_metrics:
            QMessageBox.warning(self, "No Data", "Load data into Descriptor Space first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Descriptor-Space Coordinates", "Descriptor_Space_coordinates.csv",
            "CSV Files (*.csv)")
        if not path:
            return
        try:
            import csv
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["group", "file", "kappa_star", "mean_run_length_L", "U"])
                for rec in self.descriptor_space_metrics:
                    w.writerow([rec["group"], rec["filename"], f"{rec['kappa_star']:.6f}",
                                f"{rec['run_length']:.6f}", f"{rec['U']:.6f}"])
            self.log(f"descriptor-space coordinates exported to {path}")
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")
            self.log(f"ERROR: {e}")

    def export_descriptor_space_all(self):
        """
        Exports every ensemble in the 3D Descriptor Space as one ZIP file:
        <group>/<name>.csv (chains with provenance header, plus the .meta.json
        sidecar where known), descriptor_coordinates.csv and manifest.json.
        """
        if not self.descriptor_space_data:
            QMessageBox.warning(self, "No Data", "Load or simulate ensembles first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export All Ensembles", "Descriptor_Space_ensembles.zip", "ZIP Files (*.zip)")
        if not path:
            return
        import csv
        import io
        import re
        import tempfile
        import zipfile

        def safe(text):
            text = text.replace("·", "-")
            return re.sub(r"[^\w\-. ()]+", "_", text).strip() or "ensemble"

        if self._cs_dirty or not self.descriptor_space_metrics:
            self.run_descriptor_space_analysis()
        coords = {(r["group"], r["filename"]): r for r in self.descriptor_space_metrics}
        manifest = {"program": APP_NAME, "version": APP_VERSION,
                    "engine_version": simulation_engine.ENGINE_VERSION, "ensembles": []}
        try:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z, \
                    tempfile.TemporaryDirectory() as td:
                used = set()
                for group, files in self.descriptor_space_data.items():
                    for name, polymers in files.items():
                        info = self.descriptor_space_info.get((group, name), {})
                        stem = os.path.splitext(safe(name))[0]
                        arc = f"{safe(group)}/{stem}.csv"
                        k = 2
                        while arc in used:
                            arc = f"{safe(group)}/{stem}_{k}.csv"
                            k += 1
                        used.add(arc)
                        tmp = os.path.join(td, "chains.csv")
                        meta = info.get("meta")
                        recipe_io.write_chains_csv(tmp, polymers, metadata=meta,
                                                   write_sidecar=bool(meta))
                        z.write(tmp, arc)
                        if meta:
                            z.write(recipe_io.sidecar_path(tmp), arc[:-4] + ".meta.json")
                        c = coords.get((group, name), {})
                        manifest["ensembles"].append({
                            "group": group, "name": name, "file": arc,
                            "n_chains": len(polymers), "source": info.get("source"),
                            "kappa_star": c.get("kappa_star"),
                            "mean_run_length": c.get("run_length"), "U": c.get("U")})
                buf = io.StringIO()
                w = csv.writer(buf)
                w.writerow(["group", "name", "file", "kappa_star", "mean_run_length_L", "U"])
                for e in manifest["ensembles"]:
                    w.writerow([e["group"], e["name"], e["file"]] +
                               ["" if e[k] is None else f"{e[k]:.6f}"
                                for k in ("kappa_star", "mean_run_length", "U")])
                z.writestr("descriptor_coordinates.csv", buf.getvalue())
                z.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))
            self.log(f"Exported {len(manifest['ensembles'])} ensembles to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")
            self.log(f"ERROR: {e}")

    def export_descriptor_space_html(self):
        """
        Exports the 3D descriptor space as a self-contained interactive
        Plotly HTML file — a rotatable, hoverable figure suitable for sending
        to a collaborator who doesn't run Compositional Suite.
        """
        if not self.descriptor_space_metrics:
            QMessageBox.warning(self, "No Data",
                                "Run a descriptor-space analysis before exporting.")
            return
        try:
            import plotly.graph_objects as go
        except ImportError:
            QMessageBox.critical(
                self, "Plotly Not Installed",
                "Interactive 3D export requires Plotly.\n\n"
                "Install it with:  pip install plotly",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Interactive 3D Descriptor Space", "3D_Descriptor_Space.html",
            "HTML Files (*.html)",
        )
        if not path:
            return

        try:
            # One trace per group so the legend toggles groups, matching the
            # interactive behaviour of the embedded Matplotlib panel.
            groups = {}
            for rec in self.descriptor_space_metrics:
                groups.setdefault(rec["group"], []).append(rec)

            fig = go.Figure()
            for group_name in sorted(groups):
                recs = groups[group_name]
                color = self.group_color_map.get(group_name)
                fig.add_trace(go.Scatter3d(
                    x=[r["kappa_star"] for r in recs],
                    y=[r["run_length"] for r in recs],
                    z=[r["U"] for r in recs],
                    mode="markers",
                    name=group_name,
                    marker=dict(size=5, color=color),
                    text=[r["filename"] for r in recs],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "κ*=%{x:.4f}<br>"
                        "⟨L⟩=%{y:.3f}<br>"
                        "U=%{z:.3f}<extra>" + group_name + "</extra>"
                    ),
                ))
            fig.update_layout(
                title="3D Descriptor Space",
                scene=dict(
                    xaxis_title="Neighbour correlation (κ*)",
                    yaxis_title="Mean Run Length (⟨L⟩)",
                    zaxis_title="Order, uncertainty coefficient (U)",
                ),
                legend_title_text="Ensemble Groups",
            )
            fig.write_html(path, include_plotlyjs=True)
            self.log(f"Exported interactive 3D descriptor space to:\n{path}")
            QMessageBox.information(self, "Success",
                                    "Interactive 3D figure exported.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"HTML export failed: {e}")
            self.log(f"ERROR: {e}")

    def _svg_metadata(self, kind: str) -> dict:
        """
        Provenance embedded in every exported SVG (matplotlib ``metadata``):
        program and engine version, date, and what produced the figure, as a
        JSON string in the Dublin Core ``Description`` field.
        """
        import datetime
        import json

        def default(o):
            try:
                return float(o)
            except (TypeError, ValueError):
                return str(o)

        info = {"program": APP_NAME, "version": APP_VERSION,
                "engine_version": simulation_engine.ENGINE_VERSION, "figure": kind}
        if kind == "simulator" and self.engine is not None and self.engine.block_history:
            info["synthesis"] = self._engine_global_setup()
            info["blocks"] = [dict(b) for b in self.engine.block_history]
        elif kind == "analysis":
            polymers = self.loaded_polymers or []
            info["n_chains"] = len(polymers)
            info["n_monomers"] = max((max(p) for p in polymers if p), default=0)
        elif kind == "descriptor_space":
            info["coordinates"] = self.descriptor_space_metrics
        return {
            "Title": f"{APP_NAME}: {kind} plot",
            "Creator": f"{APP_NAME} {APP_VERSION} (engine {simulation_engine.ENGINE_VERSION})",
            "Date": datetime.datetime.now().isoformat(timespec="seconds"),
            "Description": json.dumps(info, default=default),
        }

    def save_ax_as_svg(self, ax, filepath):
        extent = (ax.get_tightbbox(self.sim_fig.canvas.get_renderer())
                  .transformed(self.sim_fig.dpi_scale_trans.inverted()))
        self.sim_fig.savefig(filepath, format='svg', bbox_inches=extent,
                             metadata=self._svg_metadata("simulator"))

    def save_kde_grid_as_svg(self, filepath):
        if self.engine.N_MONs != 2:
            self.log("Cone of heterogeneity not exported: it is defined for two-monomer "
                     "copolymers only.")
            return
        temp_fig = Figure(figsize=(6, 6))
        gs = temp_fig.add_gridspec(2, 2, wspace=0.05, hspace=0.05,
                                   width_ratios=[3, 1], height_ratios=[1, 3])
        ax_main = temp_fig.add_subplot(gs[1, 0])
        ax_top = temp_fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = temp_fig.add_subplot(gs[1, 1], sharey=ax_main)
        polymer_arrays = [p.asArray() for p in self.engine.polymerArray]
        plotting.plot_2d_kde_with_marginals(
            (ax_main, ax_top, ax_right), polymer_arrays, self.engine.N_MONs
        )
        temp_fig.tight_layout(pad=1.0)
        temp_fig.savefig(filepath, format='svg', metadata=self._svg_metadata("simulator"))

    # -----------------------------------------------------------------------
    # Analysis tab
    # -----------------------------------------------------------------------

    def load_csv_file(self):
        """Prompts the user to select a CSV ensemble file and immediately runs full analysis."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Polymer Sequences CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return

        self.analysis_progress.setRange(0, 0)
        QApplication.processEvents()

        try:
            self.loaded_polymers = analysis.load_csv_data(path)
            self.loaded_filename = os.path.basename(path)
            self.loaded_file_label.setText(
                f"Loaded: {self.loaded_filename}\n({len(self.loaded_polymers)} chains)"
            )
            self.run_full_analysis()
            self.calculate_and_display_all_metrics()
            QMessageBox.information(self, "Success", "CSV file loaded and analyzed.")
            self.log(f"ANALYSIS: Loaded {os.path.basename(path)}")
        except Exception as e:
            self.loaded_polymers = None
            self.loaded_filename = ""
            self.loaded_file_label.setText("No file loaded.")
            QMessageBox.critical(self, "Error", f"Failed to load or parse CSV file:\n{e}")
            self.log(f"ERROR: Could not load {os.path.basename(path)}.")
        finally:
            self.analysis_progress.setRange(0, 100)
            self.analysis_progress.setValue(0)

    def calculate_and_display_all_metrics(self):
        """
        Runs the full analysis suite on ``self.loaded_polymers`` in a background thread
        and displays the results in a modal metrics dialog.
        """
        if not self.loaded_polymers:
            return

        # Metrics run on a background thread. If one is already in flight
        # (e.g. from rapid successive blocks) let it finish rather than
        # stomping the reference and orphaning the running thread.
        if self._analysis_worker is not None and self._analysis_worker.isRunning():
            # Re-run as soon as the in-flight pass finishes, so the metrics
            # always end up describing the newest ensemble.
            self._metrics_pending = True
            return
        self._metrics_pending = False

        self.analysis_progress.setRange(0, 0)
        QApplication.processEvents()

        self._analysis_worker = _AnalysisWorker(self.loaded_polymers)
        self._analysis_worker.finished.connect(self._on_metrics_finished)
        self._analysis_worker.error.connect(self._on_metrics_error)
        self._analysis_worker.start()

    def _on_metrics_finished(self, results: dict):
        """
        Slot called by ``_AnalysisWorker.finished``.

        Formats the analysis results as HTML, displays them in a modal dialog,
        and stores a plain-text copy for clipboard export.
        """
        try:
            pdi_data = results['pdi']
            kappa = results['kappa']
            avg_composition = results['avg_comp']
            u_val = results['U']
            mv_spread = results['mv_spread']
            spread_data = results['spread']
            cmm_results = results['cmm']
            grad_results = results['grad']
            run_length_val = results['run_length']
            n_mons = results['all_mons']
            n_chains = results['n_chains']

            # Plain-text copy buffer
            raw_text = f"Metrics for: {self.loaded_filename}\nNum Chains: {n_chains}\n\n"
            raw_text += "--- Distribution Metrics ---\n"
            raw_text += f"Avg. DP: {pdi_data['Mn_DP']:.2f}\n"
            raw_text += f"PDI (D): {pdi_data['PDI']:.3f}\n\n"
            raw_text += "--- Global Composition ---\n"
            for mid, pct in avg_composition.items():
                raw_text += f"Monomer {mid}: {pct:.2f}%\n"
            raw_text += "\n--- Descriptor Space & Sequence Metrics ---\n"
            raw_text += f"kappa* (neighbour correlation): {kappa['kappa_star']:+.4f}\n"
            raw_text += (f"kappa: {kappa['kappa']:+.4f} (P_same {kappa['p_same']:.4f}, "
                         f"chance {kappa['chance']:.4f}, P_min {kappa['p_min']:.4f})\n")
            raw_text += f"U (order, uncertainty coefficient): {u_val:.4f}\n"
            raw_text += f"Mean Run Length (<L>): {run_length_val:.4f}\n\n"
            raw_text += "--- Gradient Metrics: CMM ---\n"
            if not cmm_results:
                raw_text += "N/A\n"
            else:
                for mid, cmm in sorted(cmm_results.items()):
                    raw_text += f"Monomer {mid}: {cmm:.4f}\n"
            raw_text += "\n--- Gradient Metrics: Linear Fit ---\n(Fit: y = mx + c)\n"
            if not grad_results:
                raw_text += "N/A\n"
            else:
                for mid, (slope, r_sq) in sorted(grad_results.items()):
                    raw_text += f"Monomer {mid}: Slope = {slope:+.4f}, R2 = {r_sq:.4f}\n"
            self.last_metrics_text = raw_text

            # HTML display
            msg = (f"<b>Metrics for: {self.loaded_filename}</b><br>"
                   f"({n_chains} chains)<br><br>")
            msg += "<b>--- Distribution Metrics ---</b><br>"
            msg += f"  Avg. DP: {pdi_data['Mn_DP']:.2f}<br>"
            msg += f"  PDI (Đ): {pdi_data['PDI']:.3f}<br><br>"

            if spread_data:
                sigma = spread_data['std_dev']
                mean_pct = spread_data['mean_m1_fraction'] * 100
                msg += "<b>--- Compositional Heterogeneity ---</b><br>"
                msg += "<i>(Std Dev of Monomer Fraction — 0.0 = perfectly uniform)</i><br>"
                msg += f"  <b>Heterogeneity (σ): {sigma:.4f}</b><br>"
                msg += f"  Avg M1 Fraction: {mean_pct:.1f}%<br><br>"

            msg += "<b>--- Global Composition ---</b><br>"
            for mid, pct in avg_composition.items():
                msg += f"  Monomer {mid}: {pct:.2f}%<br>"
            msg += "<br>"

            msg += "<b>--- Descriptor Space & Sequence Metrics ---</b><br>"
            msg += (f"  κ* (neighbour correlation): {kappa['kappa_star']:+.4f} "
                    "<i>(−1 most alternating possible, 0 random, +1 blocky)</i><br>")
            msg += (f"  κ (uncorrected): {kappa['kappa']:+.4f} — P_same {kappa['p_same']:.3f}, "
                    f"chance Σf² {kappa['chance']:.3f}, P_min {kappa['p_min']:.3f}<br>")
            msg += (f"  U (order): {u_val:.4f} "
                    "<i>(0 random, 1 next unit fully predictable)</i><br>")
            msg += f"  Mean Run Length (&lt;L&gt;): {run_length_val:.4f}<br><br>"

            msg += "<b>--- Gradient Metrics: CMM ---</b><br>"
            msg += "<i>(Average normalised position 0.0–1.0)</i><br>"
            if not cmm_results:
                msg += "  N/A<br>"
            else:
                for mid, cmm in sorted(cmm_results.items()):
                    msg += f"  Monomer {mid}: {cmm:.4f}<br>"

            msg += "<br><b>--- Gradient Metrics: Linear Fit ---</b><br>"
            msg += "<i>(Fit: y = mx + c)</i><br>"
            if not grad_results:
                msg += "  N/A<br>"
            else:
                for mid, (slope, r_sq) in sorted(grad_results.items()):
                    msg += f"  Monomer {mid}: Slope = {slope:+.4f}, R² = {r_sq:.4f}<br>"

            msg += "<b>--- Global Heterogeneity ---</b><br>"
            msg += f"  Multivariate Variance: {mv_spread['total_variance']:.4f}<br>"
            msg += "  <i>(Sum of variances for all monomers)</i><br>"

            # --- SD Metrics (two-monomer copolymers only) ---
            sd_data = results.get('sd')
            msg += "<br><b>--- Structural Deviation (SD) Metrics ---</b><br>"
            raw_text += "\n--- Structural Deviation (SD) Metrics ---\n"
            if sd_data is None:
                msg += "  <i>sd_metrics package not available.</i><br>"
                raw_text += "N/A (package not available)\n"
            elif 'error' in sd_data:
                msg += f"  <i>Not computed: {sd_data['error']}</i><br>"
                raw_text += f"Not computed: {sd_data['error']}\n"
            else:
                ev = sd_data['evaluation']
                rule = ("symmetric: minimum of 4 readings" if ev.symmetric
                        else "asymmetric: minimum of 2 readings")
                header = [
                    "Conka et al. framework (Eur. Polym. J. 2023 + SI; React. Chem. Eng. 2023)",
                    f"B (minority) = monomer {ev.b_monomer}, B-Func = {ev.x_B:.1%} ({rule})",
                    f"Chains scored: {ev.n_chains_scored} of {ev.n_chains_total}, each against "
                    f"the target at its own length",
                    f"Thresholds from an ideal sample at DP {ev.reference_dp}: "
                    "Excellent ≤ ⟨SD_ideal⟩, Poor > its 98th percentile",
                ]
                msg += "".join(f"  <i>{h}</i><br>" for h in header)
                raw_text += "".join(f"{h}\n" for h in header)
                for tname, m in ev.targets.items():
                    line = (f"⟨SD⟩={m.sd_mean:.3f}, σ={m.sd_std:.3f}, "
                            f"skew={m.sd_skewness:+.2f}, CV={m.sd_cv:.2f} | thresholds "
                            f"{m.threshold_exc_good:.3f}/{m.threshold_good_poor:.3f} | "
                            f"Exc {m.frac_excellent:.0%}, Good {m.frac_good:.0%}, "
                            f"Poor {m.frac_poor:.0%}")
                    msg += f"  <b>{tname}</b>: {line}<br>"
                    raw_text += f"  {tname}: {line}\n"
                if ev.note:
                    msg += f"  <i>{ev.note}</i><br>"
                    raw_text += f"{ev.note}\n"
                if ev.scan:
                    best = ev.best_f_Gr
                    best_sd = dict(ev.scan)[best]
                    text = (f"Best-matching gradient fraction (f_Gr scan 0–1): {best:g} "
                            f"({_sd_metrics.structure_name(ev.x_B, best)}, ⟨SD⟩ = {best_sd:.3f})")
                    msg += f"  {text}<br>"
                    raw_text += f"{text}\n"
                msg += ("  <i>Skewness &lt; 0 with low CV indicates random/alternating chains; "
                        "&gt; 0 a compositional transition (RCE 2023).</i><br>")
            self._render_sd_plot(sd_data)

            self.log("--- End of Metrics Calculation ---")

            self.metrics_display.setHtml(msg)
            self.copy_metrics_btn.setEnabled(True)

        finally:
            self._analysis_worker = None
            self.analysis_progress.setRange(0, 100)
            self.analysis_progress.setValue(0)
            if getattr(self, "_metrics_pending", False):
                self._metrics_pending = False
                QTimer.singleShot(0, self.calculate_and_display_all_metrics)

    def _render_sd_plot(self, sd_data):
        """SD figure: <SD> per target, per-chain distribution, and f_Gr scan."""
        self.analysis_fig3.clear()
        if sd_data is None or 'error' in sd_data:
            ax = self.analysis_fig3.add_subplot(111)
            reason = "sd_metrics package not available" if sd_data is None else sd_data['error']
            ax.text(0.5, 0.5, f"SD metrics not computed:\n{reason}", ha='center',
                    va='center', transform=ax.transAxes, fontsize=11, wrap=True)
            ax.axis('off')
            self.analysis_canvas3.draw()
            return

        ev = sd_data['evaluation']
        names = list(ev.targets)
        gs = self.analysis_fig3.add_gridspec(1, 3, wspace=0.4)

        # Panel 1: <SD> ± sigma per target, with that target's own thresholds.
        ax1 = self.analysis_fig3.add_subplot(gs[0, 0])
        x = np.arange(len(names))
        ax1.bar(x, [ev.targets[n].sd_mean for n in names],
                yerr=[ev.targets[n].sd_std for n in names],
                color='lightsteelblue', edgecolor='steelblue', capsize=4)
        for i, n in enumerate(names):
            m = ev.targets[n]
            ax1.hlines(m.threshold_exc_good, i - 0.4, i + 0.4, colors='green', linestyles='--',
                       label='Excellent/Good' if i == 0 else None)
            ax1.hlines(m.threshold_good_poor, i - 0.4, i + 0.4, colors='orange', linestyles='--',
                       label='Good/Poor' if i == 0 else None)
        ax1.set_xticks(x)
        abbrev = {"gradient": "GD", "block-gradient": "BGD",
                  "block-gradient-block": "BGBD", "diblock": "BD"}
        ax1.set_xticklabels(
            [f"{abbrev.get(n.split(' (')[0], n.split(' (')[0])}\nf={ev.targets[n].f_Gr:.2f}"
             for n in names], fontsize=8)
        ax1.set_xlabel("target (f = gradient fraction f_Gr)", fontsize=8)
        ax1.set_ylabel('⟨SD⟩')
        ax1.set_title('SD per target\n(lower = closer match)')
        ax1.set_ylim(bottom=0)
        ax1.legend(fontsize=7)

        # Panel 2: per-chain SD distribution for the best standard target.
        best = min(ev.targets.values(), key=lambda m: m.sd_mean)
        ax2 = self.analysis_fig3.add_subplot(gs[0, 1])
        ax2.hist(best.sd_values, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
        ax2.axvline(best.threshold_exc_good, color='green', linestyle='--', linewidth=1.2,
                    label=f'Excellent/Good ({best.threshold_exc_good:.3f})')
        ax2.axvline(best.threshold_good_poor, color='orange', linestyle='--', linewidth=1.2,
                    label=f'Good/Poor ({best.threshold_good_poor:.3f})')
        ax2.set_xlabel('SD per chain')
        ax2.set_ylabel('Chains')
        ax2.set_title(f'Per-chain SD\n{best.name}')
        ax2.legend(fontsize=7)

        # Panel 3: <SD> across the target family; the minimum is the best match.
        ax3 = self.analysis_fig3.add_subplot(gs[0, 2])
        if ev.scan:
            f, v = zip(*ev.scan)
            ax3.plot(f, v, 'o-', color='purple')
            ax3.axvline(ev.best_f_Gr, color='gray', linestyle=':')
        ax3.set_xlabel('Target gradient fraction f_Gr\n(0 = diblock, 1 = gradient)')
        ax3.set_ylabel('⟨SD⟩')
        ax3.set_title('Best-matching structure')

        self.analysis_fig3.tight_layout()
        self.analysis_canvas3.draw()

    def _on_metrics_error(self, message: str):
        QMessageBox.critical(self, "Metrics Error", f"Error:\n{message}")
        self.log(f"ERROR: {message}")
        self._analysis_worker = None
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setValue(0)
        if getattr(self, "_metrics_pending", False):
            self._metrics_pending = False
            QTimer.singleShot(0, self.calculate_and_display_all_metrics)

    def copy_metrics_to_clipboard(self):
        if hasattr(self, 'last_metrics_text'):
            QApplication.clipboard().setText(self.last_metrics_text)
            self.log("Metrics copied to clipboard.")
        else:
            self.log("No metrics available to copy.")

    def run_full_analysis(self):
        """
        Renders both analysis figure panels (positional/sequence and distribution)
        for ``self.loaded_polymers``.  Called automatically after loading a CSV
        or transferring data from the simulator.
        """
        if not self.loaded_polymers:
            return
        try:
            all_mons = [m for p in self.loaded_polymers for m in p]
            if not all_mons:
                return
            n_mons = max(all_mons)

            # Fig 1: Positional
            self.analysis_fig1.clear()
            gs1 = self.analysis_fig1.add_gridspec(2, 2, hspace=0.4)
            ax_p1 = self.analysis_fig1.add_subplot(gs1[0, 0])
            ax_p2 = self.analysis_fig1.add_subplot(gs1[0, 1])
            ax_p3 = self.analysis_fig1.add_subplot(gs1[1, :])
            plotting.plot_mirrored_kde(ax_p1, self.loaded_polymers, n_mons)
            hist_data, mon_fr = analysis.get_normalized_positional_histogram_data(
                self.loaded_polymers, 60, True
            )
            plotting.plot_smoothed_positional_distribution(ax_p2, hist_data, mon_fr)
            plotting.plot_sequence_heatmap(ax_p3, self.loaded_polymers, 40, n_mons)
            self.analysis_fig1.tight_layout()
            self.analysis_canvas1.draw()

            # Fig 2: Distributions
            self.analysis_fig2.clear()
            gs2 = self.analysis_fig2.add_gridspec(2, 3, hspace=0.5, wspace=0.4)

            ax_comp_kde = self.analysis_fig2.add_subplot(gs2[0, 0:2])
            comp_data = analysis.get_composition_data(self.loaded_polymers, n_mons)
            plotting.plot_composition_kde(ax_comp_kde, comp_data)

            ax_dp = self.analysis_fig2.add_subplot(gs2[0, 2])
            plotting.plot_dp_distribution(ax_dp, self.loaded_polymers)

            inner_gs = gs2[1, 0].subgridspec(
                2, 2, wspace=0.05, hspace=0.05,
                width_ratios=[3, 1], height_ratios=[1, 3],
            )
            ax_kde_main = self.analysis_fig2.add_subplot(inner_gs[1, 0])
            ax_kde_top = self.analysis_fig2.add_subplot(inner_gs[0, 0], sharex=ax_kde_main)
            ax_kde_right = self.analysis_fig2.add_subplot(inner_gs[1, 1], sharey=ax_kde_main)
            plotting.plot_2d_kde_with_marginals(
                (ax_kde_main, ax_kde_top, ax_kde_right), self.loaded_polymers, n_mons
            )

            ax_mv = self.analysis_fig2.add_subplot(gs2[1, 1])
            plotting.plot_multivariate_parallel_coordinates(ax_mv, self.loaded_polymers, n_mons)

            ax_run = self.analysis_fig2.add_subplot(gs2[1, 2])
            plotting.plot_run_length_histogram(ax_run, self.loaded_polymers, n_mons)

            self.analysis_fig2.subplots_adjust(bottom=0.15)
            self.analysis_canvas2.draw()

        except Exception as e:
            logger.error("run_full_analysis failed: %s", e, exc_info=True)

    def _auto_update_analysis(self):
        """
        Refresh the Analysis tab from the current simulated ensemble.

        Called automatically after every block completes, so the Analysis tab
        (both plot panels and the full metric set, including SD) always reflects
        the latest simulation without the user having to transfer or recalculate
        by hand. Does not switch tabs — the Simulator stays in focus.

        Returns True if the ensemble was transferred, False if nothing met the
        DP filter (so callers can decide whether to warn).
        """
        if not self.engine or not self.engine.polymerArray:
            return False
        prune_dp = self.prune_spin.value()
        polymers = [
            p.asArray() for p in self.engine.polymerArray if p.len() >= prune_dp
        ]
        if not polymers:
            return False
        self.loaded_polymers = polymers
        self.loaded_filename = "Simulated_Transfer.csv"
        self.loaded_file_label.setText(
            f"Loaded: Internal Transfer\n({len(polymers)} chains)"
        )
        self.run_full_analysis()
        self.calculate_and_display_all_metrics()
        return True

    def transfer_sim_to_analysis(self):
        """
        Ctrl+T — refresh the Analysis tab from the current simulated ensemble
        and switch focus to it. (Refreshing now also happens automatically after
        every block; this shortcut additionally brings the tab to the front.)
        """
        if not self.engine or not self.engine.polymerArray:
            return
        if not self._auto_update_analysis():
            QMessageBox.warning(self, "Transfer Error",
                                "No chains meet the DP filter requirements.")
            return
        self.tabs.setCurrentIndex(1)
        self.log("Data transferred to Analysis tab successfully.")

    # -----------------------------------------------------------------------
    # 3D Descriptor Space tab
    # -----------------------------------------------------------------------

    def load_csv_for_descriptor_space(self):
        """
        Loads one or more CSV ensemble files into a named descriptor-space group and
        immediately re-runs the descriptor-space analysis.
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Polymer Sequences CSV", "", "CSV Files (*.csv)"
        )
        if not paths:
            return
        try:
            group_name = self.cs_group_combo.currentText()
            if group_name not in self.descriptor_space_data:
                self.descriptor_space_data[group_name] = {}
                if self.cs_group_combo.findText(group_name) == -1:
                    self.cs_group_combo.addItem(group_name)
            for path in paths:
                filename = os.path.basename(path)
                if filename not in self.descriptor_space_data[group_name]:
                    polymers = analysis.load_csv_data(path)
                    meta = None
                    side = recipe_io.sidecar_path(path)
                    if os.path.exists(side):
                        try:
                            with open(side, encoding="utf-8") as fh:
                                meta = json.load(fh)
                        except (OSError, ValueError):
                            meta = None
                    self._cs_add_entry(group_name, filename, polymers,
                                       {"source": path, "meta": meta})
            self._cs_refresh_table()
            self.run_descriptor_space_analysis()
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Failed to load or parse one or more files:\n{e}")
            self.log("ERROR: Could not load files.")

    # -- 3D Descriptor Space: ensemble bookkeeping ---------------------------
    def _cs_add_entry(self, group, name, polymers, info):
        """Adds an ensemble under a unique name in ``group``; returns the name used."""
        group = group.strip() or "Group A"
        name = name.strip() or "ensemble"
        files = self.descriptor_space_data.setdefault(group, {})
        base, k = name, 2
        while name in files:
            name = f"{base} ({k})"
            k += 1
        files[name] = polymers
        self.descriptor_space_info[(group, name)] = dict(info or {})
        if self.cs_group_combo.findText(group) == -1:
            current = self.cs_group_combo.currentText()
            items = sorted({self.cs_group_combo.itemText(i)
                            for i in range(self.cs_group_combo.count())} | {group}, key=str.lower)
            self.cs_group_combo.blockSignals(True)
            self.cs_group_combo.clear()
            self.cs_group_combo.addItems(items)
            self.cs_group_combo.setCurrentText(current)
            self.cs_group_combo.blockSignals(False)
        self._cs_dirty = True
        return name

    def _cs_take_entry(self, key):
        group, name = key
        polymers = self.descriptor_space_data[group].pop(name)
        if not self.descriptor_space_data[group]:
            del self.descriptor_space_data[group]
        self._cs_dirty = True
        return polymers, self.descriptor_space_info.pop(key, {})

    def _cs_refresh_table(self):
        t = self.cs_file_list
        t.blockSignals(True)
        t.setRowCount(0)
        for group in sorted(self.descriptor_space_data, key=str.lower):
            files = self.descriptor_space_data[group]
            for name, polymers in files.items():
                r = t.rowCount()
                t.insertRow(r)
                it_name = QTableWidgetItem(name)
                it_name.setData(Qt.UserRole, (group, name))
                it_group = QTableWidgetItem(group)
                it_n = QTableWidgetItem(str(len(polymers)))
                it_n.setFlags(it_n.flags() & ~Qt.ItemIsEditable)
                t.setItem(r, 0, it_name)
                t.setItem(r, 1, it_group)
                t.setItem(r, 2, it_n)
        t.blockSignals(False)

    def _cs_selected_keys(self):
        rows = sorted({i.row() for i in self.cs_file_list.selectedIndexes()})
        return [tuple(self.cs_file_list.item(r, 0).data(Qt.UserRole)) for r in rows]

    def _on_cs_item_changed(self, item):
        """Double-click edits: column 0 renames, column 1 moves to a (new) group."""
        row = item.row()
        key = self.cs_file_list.item(row, 0).data(Qt.UserRole)
        if key is None:
            return
        key = tuple(key)
        name = self.cs_file_list.item(row, 0).text().strip() or key[1]
        group = self.cs_file_list.item(row, 1).text().strip() or key[0]
        if (group, name) != key:
            polymers, info = self._cs_take_entry(key)
            self._cs_add_entry(group, name, polymers, info)
        self._cs_refresh_table()
        self.run_descriptor_space_analysis()

    def cs_move_selected_to_group(self):
        keys = self._cs_selected_keys()
        if not keys:
            QMessageBox.information(self, "Move to group", "Select one or more ensembles first.")
            return
        from PyQt5.QtWidgets import QInputDialog
        group, ok = QInputDialog.getItem(self, "Move to group",
                                         "Group (choose or type a new name):",
                                         list(self.descriptor_space_data.keys()), 0, True)
        if not ok or not group.strip():
            return
        for key in keys:
            polymers, info = self._cs_take_entry(key)
            self._cs_add_entry(group.strip(), key[1], polymers, info)
        self._cs_refresh_table()
        self.run_descriptor_space_analysis()

    def cs_remove_selected(self):
        for key in self._cs_selected_keys():
            self._cs_take_entry(key)
        self._cs_refresh_table()
        if self.descriptor_space_data:
            self.run_descriptor_space_analysis()
        else:
            self.clear_descriptor_space_list()

    def _cs_next_default_group(self):
        """First unused name in the series Group A, Group B, ..., Group Z, Group AA, ..."""
        used = set(self.descriptor_space_data) | {self.cs_group_combo.itemText(i)
                                            for i in range(self.cs_group_combo.count())
                                            if self.descriptor_space_data.get(self.cs_group_combo.itemText(i))}
        k = 0
        while True:
            n, label = k, ""
            while True:
                label = chr(ord("A") + n % 26) + label
                n = n // 26 - 1
                if n < 0:
                    break
            if f"Group {label}" not in used:
                return f"Group {label}"
            k += 1

    def _cs_default_run_name(self):
        """'STY/MMA · r 0.52/0.46 · feed 50:50 → 5:95' from the engine as run."""
        eng = self.engine
        setup = self._engine_global_setup()
        fmt = lambda x: f"{x:.3g}"
        parts = []
        names = setup.get("monomer_names")
        if names:
            try:
                parts.append("/".join((reactivity_library.abbreviation(m) or m)[:12] for m in names))
            except Exception:
                pass
        R = eng.R_values
        if eng.N_MONs == 2:
            parts.append(f"r {fmt(R[0][0])}/{fmt(R[1][0])}")
        else:                                   # keep names short for n >= 3
            flat = [x for row in R for x in row]
            lo, hi = min(flat), max(flat)
            parts.append(f"r all {fmt(lo)}" if lo == hi else f"r {fmt(lo)}–{fmt(hi)}")

        def feed_text(mrs):
            if len(mrs) > 2 and len({round(v, 9) for v in mrs}) == 1:
                return "equimolar"
            return ":".join(f"{v:.3g}" for v in mrs)

        feeds = [feed_text(b.get("mrs", [])) for b in eng.block_history]
        if feeds:
            parts.append("feed " + " → ".join(feeds))
        return " · ".join(parts)

    def _cs_add_simulation_run(self):
        """Adds the simulator's current ensemble to the 3D Descriptor Space."""
        if not self.cs_auto_add_check.isChecked() or self.engine is None:
            return
        if not self.engine.polymerArray:
            return
        if self.engine.block_num <= 1 or self._sim_run_group is None:
            self._sim_run_no += 1
            self._sim_run_group = self._cs_next_default_group()
        polymers = [list(p.asArray()) for p in self.engine.polymerArray]
        meta = recipe_io.build_dataset_metadata(
            self._engine_global_setup(), [dict(b) for b in self.engine.block_history],
            extra={"source": "Compositional Suite Simulator",
                   "n_blocks_completed": self.engine.block_num})
        self._cs_add_entry(self._sim_run_group, self._cs_default_run_name(), polymers,
                           {"source": "simulator", "meta": meta,
                            "run": self._sim_run_no, "block": self.engine.block_num})
        self._cs_refresh_table()
        if self.tabs.currentWidget() is self.descriptor_space_tab:
            self.run_descriptor_space_analysis()

    def clear_descriptor_space_list(self):
        self.descriptor_space_data.clear()
        self.descriptor_space_info.clear()
        self._cs_dirty = False
        self.descriptor_space_metrics = []
        self.cs_file_list.setRowCount(0)
        self.cs_results_text.clear()
        self.cs_group_combo.clear()
        self.cs_group_combo.addItem("Group A")
        self.group_color_map.clear()
        self.descriptor_space_fig.clear()
        self.descriptor_space_canvas.draw()

    def run_descriptor_space_analysis(self):
        """
        Computes (κ*, U, ⟨L⟩) for every loaded ensemble and plots them in
        a 2×2 descriptor-space scatter grid (3-D + three 2-D projections).
        """
        if not self.descriptor_space_data:
            return
        try:
            total_files = sum(len(files) for files in self.descriptor_space_data.values())
            self.cs_progress.setRange(0, total_files)
            self.cs_progress.setValue(0)
            processed_count = 0

            results_string = ""
            self.descriptor_space_fig.clear()
            gs = self.descriptor_space_fig.add_gridspec(2, 2, wspace=0.4, hspace=0.4)
            ax3d = self.descriptor_space_fig.add_subplot(gs[0, 0], projection='3d')
            ax_entropy_lambda = self.descriptor_space_fig.add_subplot(gs[0, 1])
            ax_entropy_run = self.descriptor_space_fig.add_subplot(gs[1, 0])
            ax_run_lambda = self.descriptor_space_fig.add_subplot(gs[1, 1])

            self.descriptor_space_artists_map = {}
            self.descriptor_space_metrics = []
            unique_groups = sorted(self.descriptor_space_data.keys(), key=str.lower)

            for group in unique_groups:
                if group not in self.group_color_map:
                    self.group_color_map[group] = plotting.STARK_COLORS[
                        len(self.group_color_map) % len(plotting.STARK_COLORS)
                    ]

            all_lambda, all_entropy, all_run = [], [], []

            for group_name in sorted(self.descriptor_space_data, key=str.lower):
                files = self.descriptor_space_data[group_name]
                group_color = self.group_color_map[group_name]
                group_artists = []

                for filename, polymers in files.items():
                    processed_count += 1
                    self.cs_progress.setValue(processed_count)
                    QApplication.processEvents()

                    if not polymers:
                        continue

                    info = self.descriptor_space_info.setdefault((group_name, filename), {})
                    if "desc" not in info:
                        info["desc"] = (
                            analysis.calculate_kappa_metrics(polymers)['kappa_star'],
                            analysis.calculate_uncertainty_coefficient(polymers),
                            analysis.calculate_mean_run_length(polymers))
                    lambda_val, entropy_val, run_length_val = info["desc"]

                    all_lambda.append(lambda_val)
                    all_entropy.append(entropy_val)
                    all_run.append(run_length_val)

                    self.descriptor_space_metrics.append({
                        "group": group_name,
                        "filename": filename,
                        "kappa_star": lambda_val,
                        "run_length": run_length_val,
                        "U": entropy_val,
                    })

                    results_string += (
                        f"--- [{group_name}] {filename} ---\n"
                        f"  kappa*: {lambda_val:+.4f}\n"
                        f"  U: {entropy_val:.4f}\n"
                        f"  <L>: {run_length_val:.4f}\n\n"
                    )

                    c = [group_color]
                    sc3d = ax3d.scatter(lambda_val, run_length_val, entropy_val, c=c, s=60)
                    sc2a = ax_entropy_lambda.scatter(lambda_val, entropy_val, c=c, s=50)
                    sc2b = ax_entropy_run.scatter(run_length_val, entropy_val, c=c, s=50)
                    sc2c = ax_run_lambda.scatter(lambda_val, run_length_val, c=c, s=50)
                    group_artists.extend([sc3d, sc2a, sc2b, sc2c])

                self.descriptor_space_artists_map[group_name] = group_artists

            self.cs_results_text.setText(results_string)
            self._cs_dirty = False

            def _padded(vals, pad=0.1):
                if not vals:
                    return (0, 1)
                lo, hi = min(vals), max(vals)
                margin = (hi - lo) * pad if hi != lo else (abs(hi) * pad or 0.5)
                return (lo - margin, hi + margin)

            lam_lim = (-1.05, 1.05)        # kappa* is bounded
            ent_lim = (-0.03, 1.03)        # U is bounded
            run_lim = _padded(all_run)

            xlabel = r'Neighbour correlation ($\kappa^*$)'
            ax3d.set_xlabel(xlabel)
            ax3d.set_ylabel(r'Mean Run Length ($\langle L \rangle$)')
            ax3d.set_zlabel(r'Order, uncertainty coefficient ($U$)')
            ax3d.set_xlim(*lam_lim)
            ax3d.set_ylim(*run_lim)
            ax3d.set_zlim(*ent_lim)
            ax3d.set_title('3D Descriptor Space')

            ax_entropy_lambda.set_xlabel(xlabel)
            ax_entropy_lambda.set_ylabel(r'Order, uncertainty coefficient ($U$)')
            ax_entropy_lambda.set_xlim(*lam_lim)
            ax_entropy_lambda.set_ylim(*ent_lim)
            ax_entropy_lambda.grid(True)

            ax_entropy_run.set_xlabel(r'Mean Run Length ($\langle L \rangle$)')
            ax_entropy_run.set_ylabel(r'Order, uncertainty coefficient ($U$)')
            ax_entropy_run.set_xlim(*run_lim)
            ax_entropy_run.set_ylim(*ent_lim)
            ax_entropy_run.grid(True)

            ax_run_lambda.set_xlabel(xlabel)
            ax_run_lambda.set_ylabel(r'Mean Run Length ($\langle L \rangle$)')
            ax_run_lambda.set_xlim(*lam_lim)
            ax_run_lambda.set_ylim(*run_lim)
            ax_run_lambda.grid(True)

            handles = [
                Line2D([0], [0], marker='o', color='w', label=group,
                       markerfacecolor=self.group_color_map[group], markersize=10)
                for group in unique_groups
            ]
            self.cs_legend = self.descriptor_space_fig.legend(
                handles=handles, loc='upper left', title="Ensemble Groups"
            )
            self.legend_map = {}
            for legline, legtext, group_name in zip(
                self.cs_legend.get_lines(),
                self.cs_legend.get_texts(),
                unique_groups,
            ):
                legline.set_picker(5)
                legtext.set_picker(5)
                self.legend_map[legline] = group_name
                self.legend_map[legtext] = group_name

            self.descriptor_space_fig.tight_layout()
            self.descriptor_space_canvas.draw()
            self.log("\n3D Descriptor Space analysis complete.")

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error",
                                 f"An error occurred during descriptor-space analysis:\n{e}")
            self.log(f"ERROR running descriptor-space analysis: {e}")

    def on_descriptor_space_pick(self, event):
        artist = event.artist
        if artist in self.legend_map:
            group_name = self.legend_map[artist]
            artists_to_toggle = self.descriptor_space_artists_map[group_name]
            vis = not artists_to_toggle[0].get_visible()
            for art in artists_to_toggle:
                art.set_visible(vis)
            for leg_artist, name in self.legend_map.items():
                if name == group_name:
                    leg_artist.set_alpha(1.0 if vis else 0.2)
            self.descriptor_space_canvas.draw_idle()

    # -----------------------------------------------------------------------
    # Batch simulation
    # -----------------------------------------------------------------------

    def run_batch_simulation(self):
        """
        Runs a sweep of simulations over the configured monomer-ratio scan grid.

        For each combination of monomer ratios, a fresh ``SimulationEngine`` is
        created and run, the resulting ensemble is saved as a CSV in the selected
        output directory, and summary metrics are stored in ``self.batch_results``
        for later export.  The batch runs synchronously on the UI thread;
        ``QApplication.processEvents()`` is called between runs to keep the
        progress bar responsive.
        """
        if not self.output_dir_path:
            QMessageBox.warning(self, "Warning",
                                "Please select an output directory first.")
            return

        scan_mode = self.scan_mode_combo.currentIndex()
        n_mons = self.n_mons_spin_batch.value()
        n_chains = self.n_chains_spin_batch.value()
        r_values = [[s.value() for s in row] for row in self.r_inputs_batch]
        ctr = self.ctr_spin_batch.value()
        prune_dp = self.prune_spin_batch.value()
        block_recipe = self._read_block_recipe()
        scan_block_idx = self.scan_block_combo.currentIndex()

        scan_configs = self._build_batch_scan_configs(scan_mode, n_mons)
        if scan_configs is None:
            return  # Validation failure already shown

        total_runs = len(scan_configs)
        if total_runs == 0:
            QMessageBox.warning(
                self, "Warning",
                "No valid configurations generated (check if sums > 100%).",
            )
            return

        # Recipe blocks use the headless schema ('conversion' rather than 'conv').
        blocks = [
            {"mrs": b["mrs"], "dp": b["dp"],
             "conversion": b["conv"], "purge": b["purge"]}
            for b in block_recipe
        ]
        params = {"N_MONs": n_mons, "N_CHAINs": n_chains,
                  "R_values": r_values, "Ctr": ctr}
        if self.seed_check_batch.isChecked():
            params["seed"] = self.seed_spin_batch.value()

        if not self._warn_r_matrix(r_values, self.log_batch):
            return
        lib = self._lib_state("batch")
        tasks = [
            {"params": params, "blocks": blocks, "scan_block_idx": scan_block_idx,
             "scan_mrs": list(scan_mrs), "prune_dp": prune_dp,
             "output_dir": self.output_dir_path, "index": i,
             "monomer_names": lib["names"],
             "r_sources": lib["sources"] if not lib["edited"] else None}
            for i, scan_mrs in enumerate(scan_configs)
        ]

        use_parallel = self.parallel_check.isChecked() and total_runs > 1
        mode_str = (f"parallel, {os.cpu_count()} cores" if use_parallel
                    else "sequential")

        self.batch_log_text.clear()
        self.batch_results.clear()
        self.batch_ax.clear()
        self.batch_canvas.draw()
        self.log_batch(
            f"Starting batch simulation of {total_runs} runs ({mode_str}; "
            f"{len(blocks)} block(s), scan target: Block {scan_block_idx + 1})..."
        )
        self.run_batch_btn.setEnabled(False)
        self.batch_progress_bar.setRange(0, total_runs)
        self.batch_progress_bar.setValue(0)
        QApplication.processEvents()

        results_by_index = {}
        stuck_count = 0
        done = 0

        def _consume(res):
            """Record one completed scan point's result."""
            nonlocal stuck_count
            results_by_index[res["index"]] = {
                "mrs": res["mrs"],
                "conv_hist": res["conv_hist"],
                "occurrence_hist": analysis.calculate_monomer_occurrence(
                    res["initial_amounts"], np.array(res["amounts_hist"])
                ),
            }
            stuck_count += res["statuses"].count("stuck")

        try:
            if use_parallel:
                with ProcessPoolExecutor() as executor:
                    futures = {executor.submit(headless.run_batch_point, t): t["index"]
                               for t in tasks}
                    for fut in as_completed(futures):
                        try:
                            _consume(fut.result())
                        except Exception as e:
                            self.log_batch(f"  > ERROR in run "
                                           f"{futures[fut] + 1}: {e}")
                        done += 1
                        self.batch_log_text.append(f"Completed {done}/{total_runs}")
                        self.batch_progress_bar.setValue(done)
                        QApplication.processEvents()
            else:
                for t in tasks:
                    self.log_batch(f"Run {t['index'] + 1}/{total_runs}: "
                                   f"scan MRs {[round(m, 2) for m in t['scan_mrs']]}")
                    try:
                        _consume(headless.run_batch_point(t))
                    except Exception as e:
                        self.log_batch(f"  > ERROR: {e}")
                    done += 1
                    self.batch_progress_bar.setValue(done)
                    QApplication.processEvents()
        except Exception as e:
            QMessageBox.critical(self, "Batch Error", f"Batch run failed:\n{e}")
            self.log_batch(f"ERROR: {e}")
            self.run_batch_btn.setEnabled(True)
            return

        # Re-assemble results in scan order regardless of completion order.
        self.batch_results = [results_by_index[i] for i in sorted(results_by_index)]

        summary = "\n--- Batch Simulation Complete ---"
        if stuck_count:
            summary += f"\n{stuck_count} block run(s) terminated early (stuck)."
        self.log_batch(summary)
        self.run_batch_btn.setEnabled(True)
        self.plot_batch_results()
        QMessageBox.information(self, "Success", "Batch simulation has finished.")

    def _build_batch_scan_configs(self, scan_mode: int, n_mons: int):
        """Returns a list of MR configs, or None on validation failure."""
        if scan_mode == 0:  # Linear
            start_mr = self.start_mr_spin.value()
            end_mr = self.end_mr_spin.value()
            if start_mr > end_mr:
                QMessageBox.warning(self, "Warning",
                                    "Start MR must be ≤ End MR.")
                return None
            scanned_idx = self.scan_monomer_combo.currentIndex()
            scan_mrs = np.linspace(start_mr, end_mr, self.num_steps_spin.value())
            configs = []
            for current_scan_mr in scan_mrs:
                mrs = [0.0] * n_mons
                mrs[scanned_idx] = current_scan_mr
                if n_mons - 1 > 0:
                    mr_for_others = (100.0 - current_scan_mr) / (n_mons - 1)
                    for j in range(n_mons):
                        if j != scanned_idx:
                            mrs[j] = mr_for_others
                configs.append([round(m, 1) for m in mrs])
            return configs

        else:  # Custom Grid
            ranges = []
            try:
                for i in range(n_mons - 1):
                    start = self.grid_table.cellWidget(i, 0).value()
                    end = self.grid_table.cellWidget(i, 1).value()
                    step = self.grid_table.cellWidget(i, 2).value()
                    if step <= 0:
                        QMessageBox.warning(self, "Error", "Step size must be positive.")
                        return None
                    if start > end:
                        QMessageBox.warning(
                            self, "Error", f"Monomer {i + 1}: Start must be ≤ End."
                        )
                        return None
                    ranges.append(np.arange(start, end + 0.0001, step))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error reading grid table: {e}")
                return None

            configs = []
            for combination in itertools.product(*ranges):
                remainder = 100.0 - sum(combination)
                if remainder >= -0.0001:
                    configs.append(list(combination) + [max(0.0, remainder)])
            return configs

    def plot_batch_results(self):
        self.batch_ax.clear()
        if not self.batch_results:
            self.batch_ax.text(0.5, 0.5, 'No batch data.', ha='center', va='center')
            self.batch_canvas.draw()
            return

        try:
            num_runs = len(self.batch_results)
            num_mons = self.n_mons_spin_batch.value()
            colors = plt.get_cmap('viridis', min(num_runs, 256))
            linestyles = ['-', '--', ':', '-.']

            for i, result in enumerate(self.batch_results):
                conv_hist = np.array(result['conv_hist'])
                occurrence_hist = np.nan_to_num(result['occurrence_hist'], nan=0.0)
                if len(conv_hist) < 2 or occurrence_hist.size == 0:
                    continue
                x_common = np.linspace(0, 100, 50)
                for j in range(num_mons):
                    sort_idx = np.argsort(conv_hist)
                    sorted_conv = conv_hist[sort_idx]
                    sorted_occ = occurrence_hist[sort_idx, j]
                    _, unique_indices = np.unique(sorted_conv, return_index=True)
                    if len(unique_indices) < 2:
                        continue
                    f_interp = interp1d(
                        sorted_conv[unique_indices],
                        sorted_occ[unique_indices],
                        kind='linear', bounds_error=False,
                        fill_value=(sorted_occ[0], sorted_occ[-1]),
                    )
                    self.batch_ax.plot(
                        x_common, f_interp(x_common),
                        color=colors(i % 256),
                        linestyle=linestyles[j % len(linestyles)],
                        alpha=0.7,
                    )

            self.batch_ax.set_title(
                "Batch Simulation: Normalised Monomer Occurrence (Interpolated)"
            )
            self.batch_ax.set_xlabel("Conversion (%)")
            self.batch_ax.set_ylabel("Normalised Monomer Occurrence")
            self.batch_ax.set_xlim(0, 100)
            self.batch_ax.set_ylim(0, 1.05)
            custom_lines = [
                Line2D([0], [0], color='black', linestyle=linestyles[k % len(linestyles)])
                for k in range(num_mons)
            ]
            self.batch_ax.legend(
                custom_lines, [f'Monomer {k + 1}' for k in range(num_mons)], loc='best'
            )
            self.batch_ax.grid(True)
            self.batch_fig.tight_layout()
            self.batch_canvas.draw()

        except Exception as e:
            self.batch_ax.clear()
            self.batch_ax.text(
                0.5, 0.5, f'Error plotting batch data:\n{e}',
                ha='center', va='center', color='red',
            )
            self.batch_canvas.draw()
            logger.error("plot_batch_results failed: %s", e, exc_info=True)

    def export_batch_plot_data(self):
        if not self.batch_results:
            QMessageBox.warning(self, "No Data", "No batch data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Batch Plot Data", "", "Excel Files (*.xlsx)"
        )
        if not path:
            return
        try:
            all_dfs = []
            column_labels = []
            for result in self.batch_results:
                mrs_label = "_".join(map(str, result['mrs']))
                conv_hist = result['conv_hist']
                occurrence_hist = result['occurrence_hist']
                if len(conv_hist) == 0:
                    continue
                data = {'Conversion (%)': conv_hist}
                for i in range(occurrence_hist.shape[1]):
                    data[f'Monomer {i + 1} Occurrence'] = occurrence_hist[:, i]
                all_dfs.append(pd.DataFrame(data).set_index('Conversion (%)'))
                column_labels.append(f"MRs_{mrs_label}")
            if not all_dfs:
                QMessageBox.warning(self, "No Data", "No plottable data in batch.")
                return
            pd.concat(all_dfs, axis=1, keys=column_labels).to_excel(path)
            self.log_batch(f"Exported plot data to {os.path.basename(path)}")
            QMessageBox.information(self, "Success", f"Plot data exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error",
                                 f"Failed to export data to Excel:\n{e}")
            self.log_batch(f"ERROR exporting to XLS: {e}")

    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_path = directory
            self.output_dir_lineedit.setText(directory)
            self.batch_log_text.append(f"Output directory set to: {directory}")
        else:
            self.output_dir_path = None

    def _set_ctr_controls_locked(self, locked: bool):
        """
        Locks the RAFT-agent controls once block 1 has run.

        Ctr describes the reagent, not the block, so it cannot change mid-
        synthesis; subsequent blocks inherit it and their Đ follows from it.
        """
        # The monomer set, chain count, reactivity ratios and seed are equally
        # fixed once chains exist: the engine ignores later edits, so the
        # widgets must not suggest otherwise (and exports must not read them).
        for w in (self.ctr_group_box, self.ctr_ideal_check,
                  self.target_pdi_check, self.target_pdi_spin,
                  self.n_mons_spin, self.n_chains_spin, self.seed_check,
                  self.seed_spin, self.rr_group_box):
            w.setEnabled(not locked)
        if not locked:
            self.seed_spin.setEnabled(self.seed_check.isChecked())
        if locked:
            self.ctr_group_box.setToolTip(
                "Fixed for this synthesis — the RAFT agent was chosen at "
                "block 1. Reset to change it."
            )

    def toggle_pdi_targeting(self, checked: bool):
        self.target_pdi_spin.setEnabled(checked)
        for spin in self.ctr_inputs:
            spin.setDisabled(checked or self.ctr_ideal_check.isChecked())
        self.ctr_group_box.setDisabled(checked)
        self.ctr_ideal_check.setDisabled(checked)
        self.log(
            # Solving assumes a uniform Ctr: the closed form takes one constant,
            # and collapsing a spread-out vector to an effective value
            # underestimates Đ (see effective_ctr).
            "Đ targeting enabled. A single uniform Ctr will be solved "
            "analytically and applied to every monomer."
            if checked
            else "Đ targeting disabled. The entered per-monomer Ctr will be used."
        )

    # -----------------------------------------------------------------------
    # Recipe save / load (JSON / YAML)
    # -----------------------------------------------------------------------

    @staticmethod
    def _apply_chain_count(value, spin, log):
        """
        Sets a chain-count spinbox from a recipe, reporting any clamping.

        A recipe asking for more chains than the spinbox allows would otherwise
        be silently rewritten to a different ensemble size than the file says.
        """
        requested = int(value)
        spin.setValue(requested)
        if spin.value() != requested:
            log(f"Recipe requested {requested} chains; clamped to "
                f"{spin.value()} (allowed range {_MIN_CHAINS}-{_MAX_CHAINS}).")

    @staticmethod
    def _with_library_fields(setup: dict, names, sources) -> dict:
        if names:
            setup["monomer_names"] = list(names)
        if sources:
            setup["r_sources"] = dict(sources)
        return setup

    def _simulator_global_setup(self) -> dict:
        """Synthesis setup as currently shown in the widgets (before a run)."""
        st = self._lib_state("sim")
        return self._with_library_fields({
            "n_monomers": self.n_mons_spin.value(),
            "n_chains": self.n_chains_spin.value(),
            "r_values": [[s.value() for s in row] for row in self.r_inputs],
            "ctr": recipe_io.ctr_to_json(self._current_ctr()),
            "seed": self.seed_spin.value() if self.seed_check.isChecked() else None,
            "prune_dp": self.prune_spin.value(),
        }, st["names"], st["sources"] if not st["edited"] else None)

    def _engine_global_setup(self) -> dict:
        """Synthesis setup the running engine was actually built with."""
        eng = self.engine
        run = getattr(self, "_run_setup", {}) or {}
        return self._with_library_fields({
            "n_monomers": eng.N_MONs,
            "n_chains": eng.N_CHAINs,
            "r_values": [list(row) for row in eng.R_values],
            "ctr": recipe_io.ctr_to_json(list(eng.Ctr)),
            "seed": eng.seed,
            "prune_dp": self.prune_spin.value(),
        }, run.get("monomer_names"), run.get("r_sources"))

    def _gather_simulator_recipe(self) -> dict:
        """
        Snapshots the Simulator tab into a recipe dict.

        Before any block has run, this is the widget setup with one block.
        After a run it is the synthesis as actually simulated (engine
        parameters plus every completed block), so the CLI can reproduce it;
        the current widget values are kept under ``ui.next_block``.
        """
        block = {
            "mrs": [s.value() for s in self.mr_inputs],
            "dp": self.block_len_spin.value(),
            "conversion": self.conv_spin.value(),
            "purge": not self.retain_monomers_checkbox.isChecked(),
        }
        if self.is_initialized and self.engine and self.engine.block_history:
            blocks = [{k: v for k, v in b.items() if k != "status"}
                      for b in self.engine.block_history]
            ui = {"tab": "simulator", "next_block": block,
                  "retain_monomers": self.retain_monomers_checkbox.isChecked(),
                  "show_absolute_consumption":
                      self.show_absolute_consumption_check.isChecked()}
            return recipe_io.build_recipe(self._engine_global_setup(), blocks, ui)
        ui = {
            "tab": "simulator",
            "target_pdi": {
                "enabled": self.target_pdi_check.isChecked(),
                "value": self.target_pdi_spin.value(),
            },
            "retain_monomers": self.retain_monomers_checkbox.isChecked(),
            "show_absolute_consumption": self.show_absolute_consumption_check.isChecked(),
        }
        return recipe_io.build_recipe(self._simulator_global_setup(), [block], ui)

    def _apply_simulator_recipe(self, recipe: dict):
        """Restores Simulator widget state from a recipe dict (uses its first block)."""
        g = recipe["global"]
        if int(g["n_monomers"]) > _MAX_MONOMERS:
            raise ValueError(
                f"This recipe has {g['n_monomers']} monomers; the interface supports "
                f"{_MIN_MONOMERS}–{_MAX_MONOMERS}. Run it with compositional-suite-cli instead.")
        # Setting the monomer count first rebuilds the r_inputs / mr_inputs grids.
        self.n_mons_spin.setValue(int(g["n_monomers"]))
        self._apply_chain_count(g["n_chains"], self.n_chains_spin, self.log)
        recipe = recipe_io.resolve_library_r_values(recipe)
        g = recipe["global"]
        for row_widgets, row_vals in zip(self.r_inputs, g["r_values"]):
            for spin, val in zip(row_widgets, row_vals):
                spin.setValue(float(val))
        self._set_ctr_widgets(recipe_io.ctr_from_json(g.get("ctr", _DEFAULT_CTR)))
        self.prune_spin.setValue(int(g.get("prune_dp", 0)))
        seed = g.get("seed")
        self.seed_check.setChecked(seed is not None)
        if seed is not None:
            self.seed_spin.setValue(int(seed))
        self._set_library_provenance("sim", g.get("monomer_names"), g.get("r_sources"))
        for note in recipe.get("_migration_notes", []):
            self.log(f"NOTE: {note}")

        blocks = recipe["blocks"]
        first = blocks[0]
        for spin, val in zip(self.mr_inputs, first["mrs"]):
            spin.setValue(float(val))
        self.block_len_spin.setValue(int(first["dp"]))
        self.conv_spin.setValue(float(first["conversion"]))

        ui = recipe.get("ui", {})
        tpdi = ui.get("target_pdi", {})
        self.target_pdi_check.setChecked(bool(tpdi.get("enabled", False)))
        if "value" in tpdi:
            self.target_pdi_spin.setValue(float(tpdi["value"]))
        self.retain_monomers_checkbox.setChecked(bool(ui.get("retain_monomers", False)))
        self.show_absolute_consumption_check.setChecked(
            bool(ui.get("show_absolute_consumption", False))
        )
        if len(blocks) > 1:
            self.log(f"Note: recipe defines {len(blocks)} blocks; the Simulator "
                     "loaded the first as the next-block setup. Use the Batch tab "
                     "for the full multi-block recipe.")

    def save_recipe_from_menu(self):
        """File > Save Recipe — targets the Batch tab when it is active, else the Simulator."""
        if self.tabs.currentWidget() is self.batch_tab:
            self.save_batch_recipe()
        else:
            self.save_simulator_recipe()

    def load_recipe_from_menu(self):
        """File > Load Recipe — targets the Batch tab when it is active, else the Simulator."""
        if self.tabs.currentWidget() is self.batch_tab:
            self.load_batch_recipe()
        else:
            self.load_simulator_recipe()

    def save_simulator_recipe(self):
        recipe = self._gather_simulator_recipe()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Simulator Recipe", "recipe.json",
            "Recipe Files (*.json *.yaml *.yml)",
        )
        if not path:
            return
        try:
            recipe_io.save_recipe(path, recipe)
            self.log(f"Saved recipe to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save recipe:\n{e}")

    def load_simulator_recipe(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Simulator Recipe", "",
            "Recipe Files (*.json *.yaml *.yml)",
        )
        if not path:
            return
        try:
            recipe = recipe_io.load_recipe(path)
            self._apply_simulator_recipe(recipe)
            self.log(f"Loaded recipe from {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load recipe:\n{e}")

    def _batch_global_setup(self) -> dict:
        st = self._lib_state("batch")
        return self._with_library_fields({
            "n_monomers": self.n_mons_spin_batch.value(),
            "n_chains": self.n_chains_spin_batch.value(),
            "r_values": [[s.value() for s in row] for row in self.r_inputs_batch],
            "ctr": recipe_io.ctr_to_json(self.ctr_spin_batch.value()),
            "seed": (self.seed_spin_batch.value()
                     if self.seed_check_batch.isChecked() else None),
            "prune_dp": self.prune_spin_batch.value(),
        }, st["names"], st["sources"] if not st["edited"] else None)

    def _gather_batch_recipe(self) -> dict:
        """Snapshots the Batch tab's setup, block recipe, and scan config."""
        blocks = [
            {"mrs": b["mrs"], "dp": b["dp"],
             "conversion": b["conv"], "purge": b["purge"]}
            for b in self._read_block_recipe()
        ]
        scan = {
            "mode": self.scan_mode_combo.currentIndex(),
            "block_index": self.scan_block_combo.currentIndex(),
            "monomer_index": self.scan_monomer_combo.currentIndex(),
            "start_mr": self.start_mr_spin.value(),
            "end_mr": self.end_mr_spin.value(),
            "num_steps": self.num_steps_spin.value(),
        }
        ui = {"tab": "batch", "scan": scan}
        return recipe_io.build_recipe(self._batch_global_setup(), blocks, ui)

    def _apply_batch_recipe(self, recipe: dict):
        """Restores Batch widget state (setup, block recipe table, scan config)."""
        g = recipe["global"]
        if int(g["n_monomers"]) > _MAX_MONOMERS:
            raise ValueError(
                f"This recipe has {g['n_monomers']} monomers; the interface supports "
                f"{_MIN_MONOMERS}–{_MAX_MONOMERS}. Run it with compositional-suite-cli instead.")
        self.n_mons_spin_batch.setValue(int(g["n_monomers"]))  # rebuilds grids/table
        self._apply_chain_count(g["n_chains"], self.n_chains_spin_batch,
                                self.log_batch)
        recipe = recipe_io.resolve_library_r_values(recipe)
        g = recipe["global"]
        for row_widgets, row_vals in zip(self.r_inputs_batch, g["r_values"]):
            for spin, val in zip(row_widgets, row_vals):
                spin.setValue(float(val))
        self._set_library_provenance("batch", g.get("monomer_names"), g.get("r_sources"))
        for note in recipe.get("_migration_notes", []):
            self.log_batch(f"NOTE: {note}")
        batch_ctr = recipe_io.ctr_from_json(g.get("ctr", _DEFAULT_CTR))
        # The batch tab sweeps monomer feed, not chemistry, so it carries a
        # single Ctr. A per-monomer vector is collapsed to its effective value
        # and the substitution is logged rather than applied silently.
        if isinstance(batch_ctr, list):
            collapsed = simulation_engine.effective_ctr(batch_ctr)
            self.log_batch(
                f"Recipe has a per-monomer Ctr {batch_ctr}; the batch tab uses "
                f"a single value, so the effective Ctr {collapsed:.3g} was "
                f"applied. Use the Simulator tab to keep the vector."
            )
            batch_ctr = collapsed
        # The batch tab has no ideal-exchange toggle; clamp inf to the ceiling.
        self.ctr_spin_batch.setValue(
            _MAX_CTR if not math.isfinite(batch_ctr) else batch_ctr
        )
        self.prune_spin_batch.setValue(int(g.get("prune_dp", 0)))
        seed = g.get("seed")
        self.seed_check_batch.setChecked(seed is not None)
        if seed is not None:
            self.seed_spin_batch.setValue(int(seed))

        blocks = recipe["blocks"]
        self.n_blocks_spin_batch.setValue(len(blocks))  # rebuilds the recipe table
        n_mons = self.n_mons_spin_batch.value()
        for b_idx, block in enumerate(blocks):
            for m, val in enumerate(block["mrs"]):
                w = self.block_recipe_table.cellWidget(b_idx, m)
                if w:
                    w.setValue(float(val))
            dp_w = self.block_recipe_table.cellWidget(b_idx, n_mons)
            conv_w = self.block_recipe_table.cellWidget(b_idx, n_mons + 1)
            if dp_w:
                dp_w.setValue(int(block["dp"]))
            if conv_w:
                conv_w.setValue(float(block["conversion"]))
            purge_container = self.block_recipe_table.cellWidget(b_idx, n_mons + 2)
            if purge_container:
                cb = purge_container.findChild(QCheckBox)
                if cb and cb.isEnabled():
                    cb.setChecked(bool(block.get("purge", b_idx > 0)))

        scan = recipe.get("ui", {}).get("scan", {})
        if scan:
            self.scan_mode_combo.setCurrentIndex(int(scan.get("mode", 0)))
            self._update_scan_block_combo()
            self.scan_block_combo.setCurrentIndex(
                min(int(scan.get("block_index", 0)), len(blocks) - 1)
            )
            self.scan_monomer_combo.setCurrentIndex(int(scan.get("monomer_index", 0)))
            self.start_mr_spin.setValue(float(scan.get("start_mr", 10.0)))
            self.end_mr_spin.setValue(float(scan.get("end_mr", 50.0)))
            self.num_steps_spin.setValue(int(scan.get("num_steps", 5)))

    def save_batch_recipe(self):
        recipe = self._gather_batch_recipe()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Batch Recipe", "batch_recipe.json",
            "Recipe Files (*.json *.yaml *.yml)",
        )
        if not path:
            return
        try:
            recipe_io.save_recipe(path, recipe)
            self.log_batch(f"Saved recipe to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save recipe:\n{e}")

    def load_batch_recipe(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Batch Recipe", "",
            "Recipe Files (*.json *.yaml *.yml)",
        )
        if not path:
            return
        try:
            recipe = recipe_io.load_recipe(path)
            self._apply_batch_recipe(recipe)
            self.log_batch(f"Loaded recipe from {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load recipe:\n{e}")


# ===========================================================================
# Entry point
# ===========================================================================

def app_icon_path():
    """Location of the program icon, searched the same ways the library data is found
    (installed package, frozen executable, source checkout); None if missing."""
    candidates = []
    try:
        from importlib.resources import files
        candidates.append(str(files("compositional_suite") / "data" / "icon.png"))
    except Exception:
        pass
    if getattr(sys, "_MEIPASS", None):                      # PyInstaller executable
        candidates.append(os.path.join(sys._MEIPASS, "compositional_suite", "data", "icon.png"))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "icon.png"))
    return next((c for c in candidates if os.path.exists(c)), None)


def app_icon():
    """The program icon (the logo tile), or an empty icon if the file is missing."""
    path = app_icon_path()
    return QIcon(path) if path else QIcon()


WINDOWS_APP_ID = "DTU.CompositionalSuite"


def _set_windows_app_id():
    """Only when running from source (python.exe / pythonw.exe): give the process its
    own taskbar identity so it shows our icon instead of Python's. The packaged .exe
    needs no ID: Windows takes its taskbar icon straight from the executable, and an
    explicit ID would tie the taskbar button to whatever icon Windows cached for it."""
    if sys.platform != "win32" or getattr(sys, "frozen", False):
        return False
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
        return True
    except Exception:                                       # purely cosmetic; never block start-up
        return False


def _launch_gui():
    _set_windows_app_id()
    app = QApplication(sys.argv)
    app.setWindowIcon(app_icon())
    ex = CompositionalSuiteApp()
    ex.show()
    sys.exit(app.exec_())


def main(argv=None):
    """Console-script entry point (``compositional_suite``).

    Headless mode: any CLI flag (e.g. --config exp.json --out ./run/) skips Qt
    entirely and runs the simulation from a recipe file. With no arguments,
    the GUI launches as before.
    """
    # Must run before any ProcessPoolExecutor is created, otherwise a frozen
    # (PyInstaller) build re-executes the bundle in every spawned worker.
    multiprocessing.freeze_support()

    argv = sys.argv[1:] if argv is None else argv
    if argv:
        return headless.main(argv)
    _launch_gui()


if __name__ == '__main__':
    sys.exit(main())