"""
library_dialogs.py
------------------
Qt dialogs for the bundled reactivity-ratio library.

* :class:`LibraryBrowserDialog` - searchable, read-only view of every entry
  (Help > Reactivity Ratio Library).
* :class:`FillFromLibraryDialog` - pick a library monomer for each slot, choose
  one published determination per pair, and return the oriented values so the
  caller can fill its r_ij inputs.
"""

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QCompleter, QDialog,
    QDialogButtonBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from . import reactivity_library as rl

_MAX_ROWS_SHOWN = 3000
_DATASET_LABELS = {
    "own_curated": "Own curated (shipped)",
    "textbook_table_6_1": "Textbook Table 6.1 (local)",
    "polymer_handbook": "Polymer Handbook (local)",
}
_SOURCE_NOTE = (
    "Values are reported as published; reactivity ratios depend on temperature, "
    "solvent, pH and fitting method. Cite the primary reference in published work. "
    "⚠ marks entries with data-quality flags (see docs/REACTIVITY_LIBRARY.md)."
)


def _fmt(v):
    return "" if v is None else f"{v:g}"


class LibraryBrowserDialog(QDialog):
    """Read-only, filterable table of the whole library."""

    COLUMNS = ["ID", "Monomer 1", "Monomer 2", "r1", "±r1", "r2", "±r2", "T (°C)",
               "Solvent", "Conditions", "Dataset", "Flags", "Reference", "DOI"]

    def __init__(self, parent=None, initial_text=""):
        super().__init__(parent)
        self.setWindowTitle("Reactivity Ratio Library")
        self.resize(1250, 680)
        lay = QVBoxLayout(self)

        note = QLabel(_SOURCE_NOTE)
        note.setWordWrap(True)
        lay.addWidget(note)

        bar = QHBoxLayout()
        self.filter_edit = QLineEdit(initial_text)
        self.filter_edit.setPlaceholderText("Filter by monomer name or abbreviation (e.g. 'MMA', 'acryl')")
        self.usable_check = QCheckBox("Usable by the engine only")
        self.usable_check.setChecked(True)
        self.dataset_combo = QComboBox()
        self._fill_dataset_combo()
        self.add_btn = QPushButton("Add library file…")
        self.add_btn.setToolTip("Load a local reactivity-ratio CSV in the library format "
                                "(see docs/REACTIVITY_LIBRARY.md). It is copied to "
                                + rl.user_library_dir() + " and never shipped.")
        self.add_btn.clicked.connect(self._add_file)
        bar.addWidget(self.filter_edit, 3)
        bar.addWidget(self.dataset_combo, 1)
        bar.addWidget(self.usable_check)
        bar.addWidget(self.add_btn)
        lay.addLayout(bar)

        self.sources_label = QLabel()
        self.sources_label.setWordWrap(True)
        lay.addWidget(self.sources_label)

        self.count_label = QLabel()
        lay.addWidget(self.count_label)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(self._open_doi)
        lay.addWidget(self.table)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self.filter_edit.textChanged.connect(self.refresh)
        self.usable_check.toggled.connect(self.refresh)
        self.dataset_combo.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def _fill_dataset_combo(self):
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItem("All datasets", None)
        for ds in sorted({e.dataset for e in rl.load()}, key=lambda d: rl.DATASET_PRIORITY.get(d, 9)):
            self.dataset_combo.addItem(_DATASET_LABELS.get(ds, ds), {ds})
        self.dataset_combo.blockSignals(False)

    def _add_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Add reactivity-ratio library", "",
                                              "CSV files (*.csv)")
        if not path:
            return
        try:
            dest = rl.add_library_file(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Not added", str(exc))
            return
        self._fill_dataset_combo()
        self.refresh()
        QMessageBox.information(self, "Library added", f"Copied to {dest}. It is loaded "
                                "automatically from now on.")

    def refresh(self):
        datasets = self.dataset_combo.currentData()
        self.sources_label.setText("Loaded: " + "; ".join(
            f"{name} ({n})" for name, n in rl.loaded_sources()))
        rows = rl.search(self.filter_edit.text(), usable_only=self.usable_check.isChecked(),
                         datasets=datasets)
        shown = rows[:_MAX_ROWS_SHOWN]
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(shown))
        for r, e in enumerate(shown):
            vals = [e.id, e.monomer_1, e.monomer_2, _fmt(e.r1), _fmt(e.r1_ci95),
                    _fmt(e.r2), _fmt(e.r2_ci95), e.temperature_C, e.solvent,
                    e.conditions or e.method, e.dataset, ", ".join(e.flags),
                    e.reference, e.doi]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(str(v))
                if c in (3, 4, 5, 6):
                    item.setData(Qt.EditRole, float(v) if v not in ("", None) else None)
                self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)
        more = f" (showing first {_MAX_ROWS_SHOWN}; refine the filter)" if len(rows) > len(shown) else ""
        self.count_label.setText(f"{len(rows)} entries{more}. Double-click a row to open its DOI.")
        self.table.resizeColumnsToContents()

    def _open_doi(self, row, _col):
        item = self.table.item(row, self.COLUMNS.index("DOI"))
        if item and item.text():
            QDesktopServices.openUrl(QUrl(f"https://doi.org/{item.text()}"))


class FillFromLibraryDialog(QDialog):
    """
    Choose a library monomer for each slot and one determination per pair.

    After ``exec_()`` returns Accepted, read :attr:`monomer_names` (list of
    library names) and :attr:`selection` ({(i, j): PairEntry or None}, 0-based
    i < j; None = no data / keep the current values).
    """

    def __init__(self, n_monomers: int, current_names=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fill Reactivity Ratios from Library")
        self.resize(980, 560)
        self.n = n_monomers
        self.monomer_names = []
        self.selection = {}
        self._pair_combos = {}

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Pick a monomer for each slot (type to search), then choose one published "
            "determination per pair. Multi-monomer systems use the pairwise "
            "terminal-model approximation: each r_ij comes from the binary i/j data.\n"
            + _SOURCE_NOTE)
        intro.setWordWrap(True)
        lay.addWidget(intro)

        grid = QGridLayout()
        self.name_combos = []
        display = [rl.display_name(n) for n in rl.monomers()]
        for i in range(n_monomers):
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItem("")
            combo.addItems(display)
            comp = QCompleter(display, combo)
            comp.setCaseSensitivity(Qt.CaseInsensitive)
            comp.setFilterMode(Qt.MatchContains)
            combo.setCompleter(comp)
            if current_names and i < len(current_names) and current_names[i]:
                combo.setEditText(rl.display_name(current_names[i]))
            grid.addWidget(QLabel(f"Monomer {i + 1}:"), i, 0)
            grid.addWidget(combo, i, 1)
            self.name_combos.append(combo)
        lay.addLayout(grid)

        self.lookup_btn = QPushButton("Look up pairs")
        self.lookup_btn.clicked.connect(self.lookup)
        lay.addWidget(self.lookup_btn)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        self.pair_table = QTableWidget(0, 2)
        self.pair_table.setHorizontalHeaderLabels(["Pair (r_a → first monomer)", "Determination"])
        self.pair_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.pair_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.pair_table.setSelectionMode(QAbstractItemView.NoSelection)
        lay.addWidget(self.pair_table)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Apply to reactivity ratios")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)

        if current_names and all(current_names):
            self.lookup()

    def _resolved_names(self):
        names, bad = [], []
        for i, combo in enumerate(self.name_combos):
            name = rl.resolve(combo.currentText())
            names.append(name)
            if not name:
                bad.append(i + 1)
        return names, bad

    def lookup(self):
        names, bad = self._resolved_names()
        if bad:
            self.status.setText("Not found in the library: monomer slot(s) "
                                + ", ".join(map(str, bad)) + ". Pick a name from the list.")
            self.pair_table.setRowCount(0)
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            return
        if len(set(names)) != len(names):
            self.status.setText("Each slot needs a different monomer.")
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            return
        pairs = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        self.pair_table.setRowCount(len(pairs))
        self._pair_combos = {}
        missing = 0
        for row, (i, j) in enumerate(pairs):
            label = f"M{i + 1}–M{j + 1}: {names[i]} / {names[j]}"
            self.pair_table.setItem(row, 0, QTableWidgetItem(label))
            combo = QComboBox()
            entries = rl.pair_entries(names[i], names[j])
            for pe in entries:
                combo.addItem(f"{pe.entry.id}: {pe.label()}", pe)
            if not entries:
                combo.addItem("No usable data in the library — current values kept", None)
                missing += 1
            self.pair_table.setCellWidget(row, 1, combo)
            self._pair_combos[(i, j)] = combo
        self.monomer_names = names
        msg = f"{len(pairs)} pair(s); the first option in each list is the default."
        if missing:
            msg += (f" {missing} pair(s) have no usable data: their r values stay as they are, "
                    "so set them by hand.")
        self.status.setText(msg)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)

    def _accept(self):
        self.selection = {key: combo.currentData() for key, combo in self._pair_combos.items()}
        self.accept()


def apply_selection_to_inputs(r_inputs, selection):
    """
    Writes oriented r values into a grid of r_ij spinboxes laid out like the
    engine's R_values (row i holds r_ij for j != i, in j order).

    Returns {"i-j": entry_id} for the pairs that were filled (1-based keys).
    """
    sources = {}
    for (i, j), pe in selection.items():
        if pe is None:
            continue
        r_inputs[i][j - 1].setValue(pe.r_a)   # j > i: position j-1 in row i
        r_inputs[j][i].setValue(pe.r_b)       # i < j: position i in row j
        sources[f"{i + 1}-{j + 1}"] = pe.entry.id
    return sources
