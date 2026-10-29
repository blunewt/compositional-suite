<p align="center">
  <img src="docs/images/logo.svg" alt="Compositional Suite logo" width="300">
</p>

# Compositional Suite

Stochastic copolymerisation simulation, sequence analysis, and structural-deviation
(SD) metrics — as a desktop application and a headless command-line tool.

Compositional Suite propagates chain-growth copolymerisations with user-defined reactivity
ratios, monomer feeds, and RAFT chain transfer constants (Ctr = k_tr/k_p); then
characterises the resulting ensembles positionally, compositionally, and at the
sequence level.

It follows up on *Compositional Drift*, the Monte Carlo program for predicting
copolymer composition and drift in controlled radical polymerization introduced
in A. A. A. Smith, A. Hall, V. Wu, T. Xu, *Practical Prediction of Heteropolymer
Composition and Drift*, *ACS Macro Lett.* **2019**, *8*, 36–40,
[doi:10.1021/acsmacrolett.8b00813](https://doi.org/10.1021/acsmacrolett.8b00813)
([source code](https://github.com/vince-wu/CompositionalDrift)).

---

## Install

### Download the program (no Python needed)

Get the latest version from the
[Releases page](https://github.com/blunewt/compositional-suite/releases):

| System | Download | Then |
|---|---|---|
| Windows | `CompositionalSuite.exe` | double-click to run (a single file, nothing to install; the first start can take several seconds) |
| macOS (Apple silicon) | `CompositionalSuite-macOS.zip` | unzip, move *Compositional Suite* to Applications, then right-click → *Open* the first time |

The programs are not yet code-signed. Windows may show *"Windows protected your
PC"* — click *More info → Run anyway*. macOS may refuse a double click — use
right-click → *Open* (or *System Settings → Privacy & Security → Open Anyway*).
Intel Macs and Linux: use the Python route below.

### Run with Python

Requires Python 3.10 or newer.

```bash
git clone https://github.com/blunewt/compositional-suite
cd compositional-suite
pip install -r requirements.txt
python run_compositional_suite.py
```

### Install with pip

```bash
pip install git+https://github.com/blunewt/compositional-suite
compositional-suite
```

All dependencies are installed automatically. Optional extras:

```bash
pip install "compositional-suite[all] @ git+https://github.com/blunewt/compositional-suite"
```

| Extra      | Enables                                        |
|------------|------------------------------------------------|
| `export3d` | *Export Interactive 3D (HTML)* in the 3D Descriptor Space |
| `yaml`     | `.yaml` recipe files (JSON works without it)   |
| `all`      | both of the above                              |

`compositional_suite/app.py` cannot be run as a file (the package uses relative
imports); use `run_compositional_suite.py`, `python -m compositional_suite.app`,
or the `compositional-suite` command.

---

## Usage

### Desktop application

Start it from the shortcut or app you installed, or from a terminal:

```bash
compositional-suite
```

The main window has four tabs:

1. **Simulator** — configure reactivity ratios (typed in, or filled from the
   bundled library with **From library…**), monomer feed, and the chain
   transfer constant Ctr (or a target Đ, solved for Ctr analytically); run single
   or multi-block polymerisations; watch conversion curves and sequence heatmaps
   update in real time.
2. **Analysis** — load a simulated or externally generated CSV ensemble and
   visualise positional, compositional, and sequence-level metrics, including
   **SD Metrics** (see below).
3. **3D Descriptor Space** — place multiple ensembles in a 3-D (κ*, ⟨L⟩, U)
   descriptor space, adapted from Guzman et al. so that copolymers,
   terpolymers and tetrapolymers share one scale, with three 2-D projections.
4. **Batch Simulator** — sweep one or two parameters across a grid, running a full
   simulation at each point. Runs are distributed across CPU cores.

### Headless / command line

For clusters, scripting, and reproducible runs, no GUI or display required:

```bash
compositional-suite-cli --config recipe.json --out ./run1/ --seed 7
```

| Flag       | Meaning                                              |
|------------|------------------------------------------------------|
| `--config` | Recipe file (`.json` / `.yaml`). **Required.**       |
| `--out`    | Output directory for `chains.csv` + metadata sidecar (default `./run/`) |
| `--prune`  | Override the minimum DP for export                   |
| `--seed`   | Seed the RNGs for a reproducible run                 |
| `--quiet`  | Suppress the metrics summary                         |

A sample recipe ships as
[`examples/recipes/sample_random_copolymer.json`](examples/recipes/sample_random_copolymer.json). Recipes are also written
and read by the GUI (*File → Save/Load Recipe*), so you can build a configuration
interactively and then run it at scale headlessly.

Recipes can take reactivity ratios from the library instead of numbers:

```json
"global": {"n_monomers": 2, "monomer_names": ["Styrene", "MMA"],
           "r_values": "library", "ctr": 50, "n_chains": 5000}
```

The values and library entries used are written to the output metadata. Recipes
written by engine 1.x (with `ctp`) are converted automatically, with a note.

Passing any flag to `compositional-suite` also runs headless, so `compositional-suite --config …`
is equivalent to `compositional-suite-cli --config …`.

### Troubleshooting

**`compositional-suite: command not found` (or `not recognized`)** — pip installed the
launcher into a `Scripts/` (Windows) or `bin/` (macOS/Linux) directory that is not
on your `PATH`; pip prints a warning saying so at install time. Either add that
directory to `PATH`, or skip the launcher entirely:

```bash
python -m compositional_suite.app          # GUI
python -m compositional_suite.headless --config r.json  # headless
```

**Windows warns about an unknown publisher** — the released executable is not
code-signed. Choose *More info → Run anyway*.

**Chrome says the file is dangerous or not commonly downloaded** — the same reason
(unsigned program, few downloads). Open the downloads list (Ctrl+J), click the
three dots next to the file, and choose *Keep*, then *Keep anyway*.

**Nothing happens when I start `CompositionalSuite.exe`**
- The first start can take up to a minute: the single file unpacks itself and
  Windows scans it. Check Task Manager (Details tab) for `CompositionalSuite.exe`.
- If it was downloaded as a ZIP, extract it first (right-click → *Extract All*).
- Right-click the `.exe` → *Properties* → tick *Unblock* → *OK*, then start it again.
- Windows Security → *Virus & threat protection* → *Protection history* shows whether
  Defender blocked or removed it. On Windows 11, *Smart App Control* can also block
  unsigned programs.
- If the program starts but fails, it writes the error to
  `%LOCALAPPDATA%\CompositionalSuite\crash.log` and shows a message box.

---

## Reactivity-ratio library

The program ships a curated reactivity-ratio library compiled from the primary
literature (128 determinations for 47 monomers, each with its reference). Browse it under
**Help → Reactivity Ratio Library**; fill r_ij with **From library…**; add
your own tables locally with **Add library file…**. Details:
[`docs/REACTIVITY_LIBRARY.md`](docs/REACTIVITY_LIBRARY.md).

Reactivity ratios depend on temperature, solvent, pH and fitting method — pick
the determination that matches your conditions, and cite its primary reference.

---

## Scope

Compositional Suite is designed for **ideal living (RAFT-type) copolymerization
of linear chains**:

- **Ideal polymerization only.** All chains start together and none terminate;
  termination, dead and initiator-derived chains, incomplete chain extension,
  hybrid behaviour and retardation are out of scope by design.
- **Linear chains only.** Branched, hyperbranched and crosslinked polymers,
  networks, and cyclic or bottlebrush topologies cannot be represented, and
  divinyl crosslinkers are not modelled as such. For core-first stars and side
  chains grafted from a backbone, the simulated chains can stand for individual
  arms or side chains, but not for the whole molecule.

See Help → Model Assumptions in the program.

## SD Metrics

The Analysis tab reports **structural deviation** (SD): how closely each chain
follows an idealised target architecture, following Conka et al. (*Eur. Polym.
J.* 2023 and its Supporting Information; *React. Chem. Eng.* 2023; *ACS Polym. Au*
2026). Like the binary cone of heterogeneity, SD is defined for **two-monomer
copolymers only**; other ensembles show a message instead.

- **Targets** are the papers' family indexed by the gradient fraction f_Gr:
  gradient (f_Gr = 1), block–gradient(–block) (0 < f_Gr < 1), diblock (f_Gr = 0).
  Monomer B is the minority monomer, as in the papers; the program reports which
  of your monomers that is.
- **Every chain is scored** against the target at its own length, as the minimum
  over left→right and right→left readings (and, for 50/50 compositions, with the
  labels exchanged), normalised by the B-homopolymer read right→left.
- **Quality thresholds** are derived for each target from an ideal sample of the
  same chain length and composition: Excellent ≤ ⟨SD_ideal⟩, Poor > the 98th
  percentile of SD_ideal (block-like targets use the value at f_Gr = 0.25).
- Reported per target: ⟨SD⟩, σ, skewness, CV and the Excellent/Good/Poor
  fractions; plus an f_Gr scan whose minimum names the best-matching structure.
  Negative skewness with low CV marks random or alternating chains; positive
  skewness a compositional transition.

Details, and how these values relate to the published thresholds, are in
`compositional_suite/sd_metrics/core.py`.

---

## Development

```bash
pip install -e ".[all,test]"
pytest -q                                   # 72 tests, incl. off-screen GUI tests
python tools/check_placeholders.py          # release placeholders still to fill
python tools/set_release_info.py --org <org>   # fill in the repository URLs
```

| Folder | Contents |
|---|---|
| `compositional_suite/` | the package (engine, analysis, GUI, SD metrics, library + `data/`) |
| `tests/` | pytest suite |
| `examples/recipes/` | sample recipes |
| `tools/` | release placeholder check and fill-in |
| `docs/` | sequence descriptors (`DESCRIPTORS.md`), reactivity-ratio library (`REACTIVITY_LIBRARY.md`) |

---

## Citing

If you use Compositional Suite in published work, please cite the accompanying paper.

A paper describing Compositional Suite is in preparation. Until it is published,
please cite the software itself using the metadata in [`CITATION.cff`](CITATION.cff)
(GitHub's *Cite this repository* button).

---

## License

[MIT](LICENSE). Copyright © 2026 Technical University of Denmark.

This version of the software (Compositional Suite 1.0.0) was developed by Charalampos Sideris, PhD Student, Department of Health Technology, and Anton A. A. Autzen, Associate Professor, Department of Health Technology.
