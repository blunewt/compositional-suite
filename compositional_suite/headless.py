"""
headless.py
-----------
Engine orchestration without any Qt dependency.

This module is the seam that lets the simulation run in three contexts from a
single code path:

* the GUI ("Run Batch Simulation") — imports :func:`run_batch_point` and farms
  scan points out to a ``ProcessPoolExecutor``;
* parallel workers — each subprocess imports only this module + ``recipe_io`` +
  ``simulation_engine`` (no PyQt5, pandas or scipy), keeping spawn cheap;
* the command line — ``compositional-suite-cli --config exp.json --out ./run/`` (or
  ``python -m compositional_suite.headless``) runs :func:`main` here, so HPC sweeps
  and CI never launch Qt.

Everything here is picklable / importable with no side effects.
"""

import os
import sys
import random
import argparse
import logging

import numpy as np

from . import simulation_engine
from . import recipe_io

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core multi-block runner
# ---------------------------------------------------------------------------

def run_blocks(params: dict, blocks: list):
    """
    Runs a full (potentially multi-block) polymerisation on a fresh engine.

    Args:
        params: dict accepted by ``SimulationEngine`` (N_MONs, N_CHAINs,
                R_values, Ctr).
        blocks: list of {mrs, dp, conversion, purge} dicts.

    Returns:
        (engine, statuses, block_plot) where
          * statuses   – list of "ok"/"stuck" per block
          * block_plot – list of (conv_hist, amounts_hist, initial_amounts)
                         per block, for downstream consumption plots.
    """
    eng = simulation_engine.SimulationEngine(params)
    statuses = []
    block_plot = []
    for i, b in enumerate(blocks):
        mrs = list(b["mrs"])
        dp = int(b["dp"])
        conv = float(b["conversion"])
        if i == 0:
            conv_hist, amounts_hist, status = eng.initialize_chains(mrs, dp, conv)
        else:
            conv_hist, amounts_hist, status = eng.add_new_block(
                mrs, dp, conv, purge_monomers=bool(b.get("purge", False))
            )
        statuses.append(status)
        block_plot.append(
            (conv_hist, amounts_hist, list(eng.initial_monomer_amounts_in_block))
        )
    return eng, statuses, block_plot


# ---------------------------------------------------------------------------
# Parallel batch worker (picklable, runs in a subprocess)
# ---------------------------------------------------------------------------

def run_batch_point(task: dict) -> dict:
    """
    Runs one scan point of a batch sweep and writes its CSV (with metadata).

    Designed to be dispatched through ``ProcessPoolExecutor``: ``task`` is a
    plain dict and the return value contains only picklable primitives (no
    ``Polymer`` objects, no numpy arrays beyond nested lists).

    ``task`` keys:
        params         – engine params dict
        blocks         – list of block dicts
        scan_block_idx – which block's mrs to override with scan_mrs
        scan_mrs       – the monomer ratios for this scan point
        prune_dp       – minimum DP for export
        output_dir     – where to write the CSV (or None to skip writing)
        index          – 0-based scan index (echoed back for ordering)
    """
    params = dict(task["params"])  # copy so we can mutate the seed per-task
    blocks = task["blocks"]
    scan_block_idx = task["scan_block_idx"]
    scan_mrs = list(task["scan_mrs"])
    prune_dp = task.get("prune_dp", 0)
    output_dir = task.get("output_dir")
    task_index = task.get("index", 0)

    # Derive a per-scan-point seed from the base seed so each point is
    # independently reproducible regardless of worker scheduling.
    if params.get("seed") is not None:
        params["seed"] = int(params["seed"]) + task_index

    n_mons = params["N_MONs"]
    n_chains = params["N_CHAINs"]
    ctr = params["Ctr"]
    effective_seed = params.get("seed")

    # Build the effective recipe with the scanned block's feed overridden.
    eff_blocks = []
    for i, b in enumerate(blocks):
        nb = dict(b)
        if i == scan_block_idx:
            nb["mrs"] = scan_mrs
        eff_blocks.append(nb)

    eng, statuses, block_plot = run_blocks(params, eff_blocks)

    conv_hist, amounts_hist, initial_amounts = block_plot[scan_block_idx]

    chains = [p.asArray() for p in eng.polymerArray if p.len() >= prune_dp]
    csv_path = None
    if chains and output_dir:
        mrs_tag = "_".join(f"{m:.1f}" for m in scan_mrs)
        fname = (f"NM{n_mons}_NC{n_chains}_B{len(blocks)}"
                 f"_MRs_{mrs_tag}_Ctr{recipe_io.ctr_tag(ctr)}.csv")
        csv_path = os.path.join(output_dir, fname)
        meta = recipe_io.build_dataset_metadata(
            {"n_monomers": n_mons, "n_chains": n_chains,
             "r_values": params["R_values"], "ctr": ctr,
             "seed": effective_seed, "prune_dp": prune_dp,
             "monomer_names": task.get("monomer_names"),
             "r_sources": task.get("r_sources")},
            [dict(b) for b in eng.block_history],
            extra={"scan_block": scan_block_idx + 1,
                   "scan_mrs": [round(m, 2) for m in scan_mrs]},
        )
        recipe_io.write_chains_csv(csv_path, chains, metadata=meta)

    return {
        "index": task.get("index", 0),
        "mrs": [round(m, 2) for m in scan_mrs],
        "conv_hist": list(conv_hist),
        "amounts_hist": np.asarray(amounts_hist).tolist(),
        "initial_amounts": list(initial_amounts),
        "statuses": statuses,
        "n_exported": len(chains),
        "csv_path": csv_path,
    }


# ---------------------------------------------------------------------------
# Single-recipe headless run (CLI)
# ---------------------------------------------------------------------------

def run_from_recipe(recipe: dict, out_dir: str, prune_dp: int = None,
                    seed: int = None):
    """
    Runs a single ensemble described by ``recipe`` and writes it to ``out_dir``.

    The seed precedence is: ``--seed`` CLI override > recipe ``global.seed`` >
    no seeding. The effective seed is recorded in the dataset metadata.

    Returns (engine, statuses, chains, csv_path).
    """
    recipe = recipe_io.resolve_library_r_values(recipe)
    params = recipe_io.engine_params_from_recipe(recipe)
    if seed is not None:
        params["seed"] = int(seed)
    effective_seed = params.get("seed")

    blocks = recipe["blocks"]
    eng, statuses, _ = run_blocks(params, blocks)

    g = recipe["global"]
    prune = g.get("prune_dp", 0) if prune_dp is None else prune_dp
    chains = [p.asArray() for p in eng.polymerArray if p.len() >= prune]

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "chains.csv")
    meta = recipe_io.build_dataset_metadata(
        {"n_monomers": params["N_MONs"], "n_chains": params["N_CHAINs"],
         "r_values": params["R_values"], "ctr": params["Ctr"],
         "seed": effective_seed, "prune_dp": prune,
         "monomer_names": g.get("monomer_names"), "r_sources": g.get("r_sources")},
        eng.block_history,
    )
    recipe_io.write_chains_csv(csv_path, chains, metadata=meta)
    return eng, statuses, chains, csv_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """
    Headless command-line entry point.

    Example:
        compositional-suite-cli --config exp.json --out ./run1/ --seed 7
    """
    parser = argparse.ArgumentParser(
        prog="compositional-suite-cli",
        description="Run a Compositional Suite simulation headlessly from a recipe file "
                    "(JSON or YAML). No GUI is launched.",
    )
    parser.add_argument("--config", required=True,
                        help="Path to a recipe file (.json / .yaml).")
    parser.add_argument("--out", default="./run/",
                        help="Output directory for chains.csv + metadata "
                             "(default: ./run/).")
    parser.add_argument("--prune", type=int, default=None,
                        help="Override the minimum DP for export.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the RNGs for a reproducible run.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress the metrics summary.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s  %(message)s")

    try:
        recipe = recipe_io.load_recipe(args.config)
    except Exception as exc:
        print(f"ERROR: could not load recipe '{args.config}': {exc}", file=sys.stderr)
        return 2
    for note in recipe.get("_migration_notes", []):
        print(f"NOTE: {note}", file=sys.stderr)

    try:
        recipe = recipe_io.resolve_library_r_values(recipe)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for warning in simulation_engine.check_r_values(recipe["global"]["r_values"]):
        print(f"WARNING: {warning}", file=sys.stderr)

    eng, statuses, chains, csv_path = run_from_recipe(
        recipe, args.out, prune_dp=args.prune, seed=args.seed
    )

    if any(s == "stuck" for s in statuses):
        print(f"WARNING: {statuses.count('stuck')} block(s) terminated early "
              f"(stuck).", file=sys.stderr)

    print(f"Wrote {len(chains)} chains to {csv_path}")
    print(f"Metadata sidecar: {recipe_io.sidecar_path(csv_path)}")

    if not args.quiet and chains:
        # analysis is only imported here (main process), keeping workers light.
        from . import analysis
        pdi = analysis.calculate_pdi(chains)
        comp = analysis.calculate_average_composition(chains)
        comp_str = ", ".join(f"M{k}: {v:.1f}%" for k, v in comp.items())
        print(f"  Mn(DP)={pdi['Mn_DP']:.2f}  PDI={pdi['PDI']:.3f}")
        print(f"  Composition: {comp_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
