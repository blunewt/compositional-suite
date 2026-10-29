"""
recipe_io.py
------------
Serialisation layer shared by the GUI, the headless CLI, and the parallel
batch workers.  Deliberately depends only on the standard library (plus the
``ENGINE_VERSION`` string), so it is cheap to import inside ``ProcessPool``
workers and never pulls in PyQt5 / pandas / scipy.

Responsibilities
~~~~~~~~~~~~~~~~
* **Recipe schema** – a single JSON/YAML structure describing an experiment
  (global setup + one or more blocks + optional UI-only extras).  Used by the
  "Save / Load recipe" buttons and by the ``--config`` CLI flag.
* **Dataset metadata** – the provenance block stamped onto every exported CSV
  (commented ``# key=value`` header) and written alongside it as a sidecar
  ``.meta.json`` file, so a chain CSV can never be orphaned from the conditions
  that produced it.
* **Robust CSV reading** – the canonical chain-sequence reader, tolerant of the
  metadata header comments and blank lines we now emit.

Recipe schema (version 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: json

    {
      "compositional_suite_recipe": 2,
      "engine_version": "2.0.0",
      "global": {
        "n_monomers": 2,
        "n_chains": 1000,
        "r_values": [[1.0], [1.0]],
        "ctr": 10.0,
        "prune_dp": 0,
        "monomer_names": ["Styrene", "Methyl methacrylate"],
        "r_sources": {"1-2": "OWN-0003"}
      },
      "blocks": [
        {"mrs": [50.0, 50.0], "dp": 60, "conversion": 100.0, "purge": false}
      ],
      "ui": { "...": "optional GUI-only state, ignored by the engine/CLI" }
    }

``global.ctr`` is the chain transfer constant k_tr/k_p. It may be a single
value (broadcast to every monomer) or one value per monomer, indexed by the
terminal unit carrying the radical — e.g. ``[10.0, 200.0]``. The string
``"ideal"`` stands for the instantaneous-exchange limit (infinity), which has
no portable JSON spelling, and is accepted in either position.

``monomer_names`` and ``r_sources`` are optional. ``r_sources`` maps a 1-based
pair ("1-2") to the reactivity-ratio library entry the values came from.
``r_values`` may also be the string ``"library"``: the matrix is then built
from ``monomer_names`` with the library's default entry for every pair.

Version history
~~~~~~~~~~~~~~~
* 1 - engine 1.x; dispersity set by a unitless chain-transfer probability
  ``ctp`` in (0, 1]. Loaded transparently by :func:`migrate_recipe`.
* 2 - engine 2.x; physical chain transfer constant ``ctr`` = k_tr/k_p.
  (Some engine-2.0 files were written with the number still at 1; they are
  recognised by carrying ``ctr`` and are accepted as version 2.)
"""

import os
import csv
import json
import math
import datetime

from .simulation_engine import ENGINE_VERSION, IDEAL_EXCHANGE

RECIPE_FORMAT_VERSION = 2

# Order of the compact comment header written to the top of every CSV export.
_HEADER_KEYS = (
    "engine_version", "date", "n_monomers", "monomer_names", "n_chains",
    "r_ij", "r_sources", "Ctr", "seed", "prune_dp", "blocks",
)

# Infinity has no portable JSON/YAML spelling, so the ideal-exchange limit is
# serialised as this sentinel string instead.
_CTR_IDEAL_TOKEN = "ideal"


def _ctr_scalar_to_json(ctr):
    return _CTR_IDEAL_TOKEN if not math.isfinite(float(ctr)) else float(ctr)


def _ctr_scalar_from_json(value) -> float:
    if isinstance(value, str):
        if value.strip().lower() in (_CTR_IDEAL_TOKEN, "inf", "infinity"):
            return IDEAL_EXCHANGE
        raise ValueError(
            f"ctr must be a positive number or '{_CTR_IDEAL_TOKEN}', "
            f"got {value!r}."
        )
    return float(value)


def ctr_to_json(ctr):
    """
    Serialises a chain transfer constant (inf -> ``"ideal"``).

    Handles both a scalar and a per-monomer list, preserving the shape so a
    round-trip through disk cannot silently collapse a vector to a scalar.
    """
    if ctr is None:
        return None
    if isinstance(ctr, (list, tuple)):
        return [_ctr_scalar_to_json(c) for c in ctr]
    return _ctr_scalar_to_json(ctr)


def ctr_from_json(value):
    """
    Parses a serialised chain transfer constant.

    Accepts a positive number, ``"ideal"`` / ``"inf"`` for the
    instantaneous-exchange limit, or a list of either (one per monomer).
    """
    if isinstance(value, (list, tuple)):
        return [_ctr_scalar_from_json(v) for v in value]
    return _ctr_scalar_from_json(value)


def ctr_tag(ctr) -> str:
    """
    Filename-safe rendering of a chain transfer constant.

    Scalars render as "12p5" / "ideal"; per-monomer vectors join entries with
    "-" (e.g. "10-200"), collapsing to one token when all entries agree.
    """
    if ctr is None:
        return _CTR_IDEAL_TOKEN
    if isinstance(ctr, (list, tuple)):
        parts = [ctr_tag(c) for c in ctr]
        return parts[0] if len(set(parts)) == 1 else "-".join(parts)
    if not math.isfinite(float(ctr)):
        return _CTR_IDEAL_TOKEN
    return f"{float(ctr):g}".replace(".", "p").replace("+", "")


# ---------------------------------------------------------------------------
# Recipe build / validate
# ---------------------------------------------------------------------------

def build_recipe(global_setup: dict, blocks: list, ui: dict = None) -> dict:
    """
    Assembles a recipe dict from a global-setup dict and a list of block dicts.

    Args:
        global_setup: keys n_monomers, n_chains, r_values, ctr, prune_dp.
        blocks:       list of {mrs, dp, conversion, purge} dicts.
        ui:           optional GUI-only extras (tab, scan config, target PDI).
    """
    recipe = {
        "compositional_suite_recipe": RECIPE_FORMAT_VERSION,
        "engine_version": ENGINE_VERSION,
        "global": dict(global_setup),
        "blocks": [dict(b) for b in blocks],
    }
    if ui:
        recipe["ui"] = ui
    return recipe


def migrate_recipe(recipe: dict):
    """
    Upgrades an older recipe to the current schema in memory.

    Returns ``(recipe, notes)``; ``notes`` is a list of human-readable
    messages describing every change made (empty if none was needed).

    Version 1 (engine 1.x) recipes stored ``ctp``; the old engine turned it
    into a transfer constant per block as ``C = block_len * ctp / (1 - ctp)``
    (``ctp == 1`` meant one monomer per step, i.e. ideal exchange). That
    mapping is reproduced exactly for block 1. Because the old engine
    re-derived C from each block's length, multi-block recipes whose blocks
    differ in length cannot be reproduced exactly by a single Ctr; a note
    says so.
    """
    if not isinstance(recipe, dict):
        return recipe, []
    notes = []
    version = recipe.get("compositional_suite_recipe", 1)
    if isinstance(version, int) and version > RECIPE_FORMAT_VERSION:
        raise ValueError(
            f"Recipe format {version} is newer than this program supports "
            f"({RECIPE_FORMAT_VERSION}). Update the program to load it.")
    g = recipe.get("global")
    if isinstance(g, dict) and "ctr" not in g and "ctp" in g:
        blocks = recipe.get("blocks") or [{}]
        dps = [b.get("dp") for b in blocks if isinstance(b, dict)]
        try:
            ctp = float(g["ctp"])
            dp1 = float(dps[0])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"Cannot migrate legacy 'ctp' recipe: {exc}") from exc
        if not 0 < ctp <= 1:
            raise ValueError(f"Legacy ctp must be in (0, 1], got {ctp}.")
        ctr = IDEAL_EXCHANGE if ctp >= 1.0 else dp1 * ctp / (1.0 - ctp)
        g = dict(g)
        g.pop("ctp")
        g["ctr"] = ctr_to_json(ctr)
        recipe = dict(recipe, **{"global": g})
        notes.append(
            f"Legacy recipe (format 1, CTP = {ctp:g}) converted to Ctr = "
            f"{'ideal' if not math.isfinite(ctr) else format(ctr, '.4g')} "
            f"using block 1 (DP = {dp1:g}).")
        if len({d for d in dps if d is not None}) > 1:
            notes.append(
                "The old engine re-derived the transfer constant from each "
                "block's length; with blocks of different DP, later blocks will "
                "not reproduce the original run exactly.")
    if recipe.get("compositional_suite_recipe") != RECIPE_FORMAT_VERSION:
        recipe = dict(recipe, compositional_suite_recipe=RECIPE_FORMAT_VERSION)
    return recipe, notes


def validate_recipe(recipe: dict) -> dict:
    """
    Validates the structure of a loaded recipe, raising ValueError with a
    human-readable message on the first problem found.  Returns the recipe
    unchanged on success so callers can write ``r = validate_recipe(load(...))``.
    """
    if not isinstance(recipe, dict):
        raise ValueError("Recipe must be a JSON/YAML object.")

    g = recipe.get("global")
    if not isinstance(g, dict):
        raise ValueError("Recipe is missing a 'global' section.")

    n = g.get("n_monomers")
    if not isinstance(n, int) or n < 2:
        raise ValueError("global.n_monomers must be an integer >= 2.")

    if not isinstance(g.get("n_chains"), int) or g["n_chains"] < 1:
        raise ValueError("global.n_chains must be a positive integer.")

    names = g.get("monomer_names")
    if names is not None:
        if (not isinstance(names, list) or len(names) != n
                or not all(isinstance(x, str) for x in names)):
            raise ValueError(f"global.monomer_names must be a list of {n} strings.")
    if g.get("r_sources") is not None and not isinstance(g["r_sources"], dict):
        raise ValueError("global.r_sources must be an object such as {\"1-2\": \"OWN-0003\"}.")

    r = g.get("r_values")
    if r == "library":
        if names is None:
            raise ValueError("global.r_values = \"library\" needs global.monomer_names.")
    else:
        if not isinstance(r, list) or len(r) != n:
            raise ValueError(f"global.r_values must have {n} rows (one per monomer).")
        for i, row in enumerate(r):
            if not isinstance(row, list) or len(row) != n - 1:
                raise ValueError(
                    f"global.r_values[{i}] must have {n - 1} entries "
                    f"(cross ratios, excluding the diagonal)."
                )
            for v in row:
                if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v) or v < 0:
                    raise ValueError(
                        f"global.r_values[{i}] contains {v!r}; reactivity ratios "
                        "must be numbers >= 0.")

    if "ctr" not in g:
        raise ValueError(
            "global.ctr (chain transfer constant k_tr/k_p) is required."
        )
    try:
        ctr = ctr_from_json(g["ctr"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"global.ctr is invalid: {exc}") from exc
    ctr_list = ctr if isinstance(ctr, list) else [ctr]
    if isinstance(ctr, list) and len(ctr) != n:
        raise ValueError(
            f"global.ctr must be a single value or one per monomer ({n}), "
            f"got {len(ctr)}."
        )
    for i, c in enumerate(ctr_list):
        if math.isnan(c) or c <= 0:
            raise ValueError(
                f"global.ctr[{i}] must be a positive number, got {c!r}."
            )

    blocks = recipe.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Recipe must contain at least one block.")
    for i, b in enumerate(blocks):
        if not isinstance(b, dict):
            raise ValueError(f"blocks[{i}] must be an object.")
        mrs = b.get("mrs")
        if not isinstance(mrs, list) or len(mrs) != n:
            raise ValueError(f"blocks[{i}].mrs must have {n} monomer ratios.")
        if "dp" not in b or "conversion" not in b:
            raise ValueError(f"blocks[{i}] must define 'dp' and 'conversion'.")
        if not all(isinstance(m, (int, float)) and m >= 0 for m in mrs) or sum(mrs) <= 0:
            raise ValueError(f"blocks[{i}].mrs must be numbers >= 0, not all zero.")
        if not (isinstance(b["dp"], (int, float)) and b["dp"] >= 1):
            raise ValueError(f"blocks[{i}].dp must be >= 1.")
        if not (isinstance(b["conversion"], (int, float)) and 0 < b["conversion"] <= 100):
            raise ValueError(f"blocks[{i}].conversion must be in (0, 100].")

    return recipe


def resolve_library_r_values(recipe: dict) -> dict:
    """
    If ``global.r_values`` is ``"library"``, builds the matrix from
    ``monomer_names`` and records the entries used in ``r_sources``.
    Returns the (possibly updated) recipe; raises ValueError on missing data.
    """
    g = recipe["global"]
    if g.get("r_values") != "library":
        return recipe
    from . import reactivity_library as rl   # lazy: only when asked for
    names = []
    for raw in g["monomer_names"]:
        name = rl.resolve(raw)
        if not name:
            raise ValueError(f"Monomer {raw!r} is not in the reactivity-ratio library.")
        names.append(name)
    r_values, sources, missing = rl.build_r_matrix(names)
    if missing:
        pairs = ", ".join(f"{names[i - 1]}/{names[j - 1]}" for i, j in missing)
        raise ValueError(f"No usable library data for: {pairs}. Give r_values explicitly.")
    g = dict(g, r_values=r_values, monomer_names=names, r_sources=sources)
    return dict(recipe, **{"global": g})


def engine_params_from_recipe(recipe: dict) -> dict:
    """Extracts the dict expected by ``SimulationEngine(params)`` from a recipe."""
    recipe = resolve_library_r_values(recipe)
    g = recipe["global"]
    params = {
        "N_MONs": g["n_monomers"],
        "N_CHAINs": g["n_chains"],
        "R_values": g["r_values"],
        "Ctr": ctr_from_json(g["ctr"]),
    }
    if g.get("seed") is not None:
        params["seed"] = int(g["seed"])
    return params


# ---------------------------------------------------------------------------
# Recipe save / load (JSON + optional YAML)
# ---------------------------------------------------------------------------

def _is_yaml(path: str) -> bool:
    return path.lower().endswith((".yaml", ".yml"))


def save_recipe(path: str, recipe: dict) -> None:
    """Writes a recipe to disk as JSON (.json) or YAML (.yaml/.yml)."""
    recipe = {k: v for k, v in recipe.items() if not k.startswith("_")}
    # Infinity has no JSON spelling ("Infinity" is a Python extension other
    # tools reject), so the ideal-exchange limit is always written as "ideal".
    g = recipe.get("global")
    if isinstance(g, dict) and g.get("ctr") is not None:
        recipe = dict(recipe, **{"global": dict(g, ctr=ctr_to_json(ctr_from_json(g["ctr"])))})
    if _is_yaml(path):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "Saving YAML requires PyYAML (`pip install pyyaml`). "
                "Use a .json filename instead."
            ) from exc
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(recipe, f, sort_keys=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recipe, f, indent=2, allow_nan=False)


def load_recipe(path: str) -> dict:
    """
    Loads, migrates and validates a recipe from a JSON or YAML file.

    Migration notes (see :func:`migrate_recipe`) are attached under the
    transient key ``"_migration_notes"`` so callers can show them.
    """
    if _is_yaml(path):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "Loading YAML requires PyYAML (`pip install pyyaml`). "
                "Convert the file to .json or install PyYAML."
            ) from exc
        with open(path, "r", encoding="utf-8") as f:
            recipe = yaml.safe_load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            recipe = json.load(f)
    recipe, notes = migrate_recipe(recipe)
    recipe = validate_recipe(recipe)
    if notes:
        recipe = dict(recipe, _migration_notes=notes)
    return recipe


# ---------------------------------------------------------------------------
# Dataset metadata (CSV header + sidecar JSON)
# ---------------------------------------------------------------------------

def build_dataset_metadata(global_setup: dict, blocks: list,
                           extra: dict = None) -> dict:
    """
    Builds the provenance metadata stamped onto an exported chain CSV.

    The returned dict is both serialised as the sidecar ``.meta.json`` and
    rendered into the commented header at the top of the CSV.
    """
    g = global_setup
    meta = {
        "format": "compositional_suite-dataset",
        "engine_version": ENGINE_VERSION,
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_monomers": g.get("n_monomers"),
        "n_chains": g.get("n_chains"),
        "r_ij": g.get("r_values"),
        "Ctr": ctr_to_json(g.get("ctr")),
        "seed": g.get("seed"),
        "prune_dp": g.get("prune_dp"),
        "blocks": [dict(b) for b in blocks],
    }
    if g.get("monomer_names"):
        meta["monomer_names"] = list(g["monomer_names"])
    if g.get("r_sources"):
        meta["r_sources"] = dict(g["r_sources"])
    if extra:
        meta.update(extra)
    return meta


def metadata_header_lines(meta: dict) -> list:
    """Renders dataset metadata as a list of ``# key=value`` comment lines."""
    lines = [
        "# Compositional Suite chain dataset",
        "# Provenance below; full recipe in the accompanying .meta.json sidecar.",
    ]
    for key in _HEADER_KEYS:
        if key in meta and meta[key] is not None:
            lines.append(f"# {key}={json.dumps(meta[key])}")
    return lines


def sidecar_path(csv_path: str) -> str:
    """Returns the sidecar metadata path for a given CSV path."""
    return os.path.splitext(csv_path)[0] + ".meta.json"


def write_chains_csv(path: str, chains: list, metadata: dict = None,
                     write_sidecar: bool = True) -> None:
    """
    Writes polymer chains (one chain per row) to ``path``.

    If ``metadata`` is provided, a commented ``# key=value`` provenance header is
    prepended and — when ``write_sidecar`` is True — a ``.meta.json`` file is
    written next to the CSV.  Comment lines are transparently skipped by
    :func:`read_chains_csv` / ``analysis.load_csv_data``.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        if metadata:
            for line in metadata_header_lines(metadata):
                f.write(line + "\n")
        csv.writer(f).writerows(chains)

    if metadata and write_sidecar:
        with open(sidecar_path(path), "w", encoding="utf-8") as jf:
            json.dump(metadata, jf, indent=2)


# ---------------------------------------------------------------------------
# Robust CSV reading
# ---------------------------------------------------------------------------

def read_chains_csv(filepath: str) -> list:
    """
    Reads polymer sequences from a CSV file, one chain per row.

    Tolerant of the provenance header we now emit: blank rows and rows whose
    first cell begins with ``#`` are skipped.  Remaining cells are parsed as
    integer (1-indexed) monomer IDs.
    """
    sequences = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].lstrip().startswith("#"):
                continue
            chain = [int(m) for m in row if m.strip()]
            if chain:
                sequences.append(chain)
    return sequences
