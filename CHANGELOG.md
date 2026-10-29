# Changelog

## 1.0.0 — first public release

Simulation
- Kinetic Monte Carlo simulation of ideal living (RAFT-type) copolymerization
  in the terminal model, for any number of monomers (2–8 in the interface).
- Multi-block synthesis with exact degree-of-polymerization stoichiometry and
  optional removal of leftover monomer between blocks.
- Dispersity set by a target value (default) or a chain transfer constant
  C_tr; C_tr is obtained analytically from the target dispersity.
- Recipes (JSON/YAML) and a headless command-line runner for scripted and
  cluster use; fixed seeds make runs reproducible.

Reactivity ratios
- Built-in curated library of 128 reactivity-ratio determinations (47
  monomers) from the primary literature; a default entry is proposed per
  monomer pair and every entry can be chosen in the library browser. Local
  tables can be added.

Analysis
- Composition, chain-length distribution, M_n and Đ; consumption and drift
  plots; positional monomer density; sequence heat maps.
- Sequence descriptors κ* (neighbour correlation), U (order) and ⟨L⟩ (mean
  run length), comparable across numbers of monomers (docs/DESCRIPTORS.md).
- Structural deviation against gradient, block–gradient, block–gradient–block
  and diblock targets after Conka et al.; cone of heterogeneity (two-monomer
  systems).
- 3D Descriptor Space tab: every simulated block is added automatically;
  ensembles can be renamed, regrouped and exported together (ZIP).
- Batch Simulator for parameter sweeps, run in parallel.

Exports and reproducibility
- Chain CSV files with a provenance header and JSON sidecar; SVG figures with
  embedded metadata; interactive 3D HTML (optional plotly).

Installation
- DTU copyright notice in the LICENSE, the README, the About dialog and the
  Windows executable's file properties (with product name and version).
- The Windows executable reports start-up errors in a message box and in
  `%LOCALAPPDATA%\CompositionalSuite\crash.log` instead of failing silently.
- The Windows icon file uses the classic layout (bitmap frames below 256 px,
  PNG at 256 px); the build log reports whether the icon was embedded.
- Program icon and logo: window and taskbar icon, icons embedded in the Windows
  executable and the macOS app, logo in the README.
- A standalone Windows executable and a macOS application (Apple silicon),
  built by the CI workflow and attached to each release;
  run from a checkout with `pip install -r requirements.txt`; pip install.
