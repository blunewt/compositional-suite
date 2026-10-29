"""
analysis.py
-----------
Statistical analysis functions for polymer chain ensembles.

Organised into the following sections:

* **Internal helpers** – shared utilities (_build_positional_records, _chain_counter)
* **Compositional spread** – heterogeneity of monomer fractions across chains
* **Composition fit metrics** – R² / slope and fractional std for composition vs DP
* **Basic composition statistics** – average composition of the ensemble
* **Positional / local composition data** – normalised-position binning utilities
* **File I/O** – CSV loading
* **Sequence statistics** – κ*, U, run lengths, PDI, monomer occurrence
"""

import logging
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import linregress, gaussian_kde

from . import recipe_io

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_positional_records(polymer_array: list) -> pd.DataFrame:
    """
    Shared helper: builds a flat DataFrame of (monomer_id, normalized_position)
    from all chains.  Extracted to eliminate the duplicated boilerplate that
    previously appeared in get_local_composition_data, get_normalized_positional_data,
    and get_normalized_positional_histogram_data.
    """
    records = []
    for chain in polymer_array:
        chain_len = len(chain)
        if chain_len > 1:
            for i, monomer_id in enumerate(chain):
                records.append({
                    'monomer_id': monomer_id,
                    'normalized_position': i / (chain_len - 1),
                })
    if not records:
        return pd.DataFrame(columns=['monomer_id', 'normalized_position'])
    return pd.DataFrame(records)


def _chain_counter(chain: list) -> Counter:
    """Returns a Counter for one chain. Used to avoid repeated O(n) .count() calls."""
    return Counter(chain)


# ---------------------------------------------------------------------------
# Compositional spread
# ---------------------------------------------------------------------------

def calculate_compositional_spread(polymer_array: list) -> dict:
    """
    Calculates the spread (heterogeneity) of the monomer-1 fraction across chains.
    Returns mean fraction, std dev, and chain count.
    0.0 std dev = perfectly uniform. Higher = more heterogeneous.
    """
    if not polymer_array:
        return {}

    fractions = []
    for chain in polymer_array:
        counts = _chain_counter(chain)
        n1 = counts[1]
        total = sum(counts.values())
        if total > 0:
            fractions.append(n1 / total)

    if not fractions:
        return {}

    arr = np.array(fractions)
    return {
        'mean_m1_fraction': float(np.mean(arr)),
        'std_dev': float(np.std(arr)),
        'count': len(arr),
    }


# ---------------------------------------------------------------------------
# Composition fit metrics
# ---------------------------------------------------------------------------

def calculate_ideal_fit_r2(x_values, y_values):
    """
    Calculates R² against the ideal line whose slope equals the global average
    ratio (sum(y) / sum(x)).  Sensitive to PDI / molecular weight differences.
    """
    x = np.array(x_values)
    y = np.array(y_values)

    if len(x) < 2 or np.sum(x) == 0:
        return 0.0, 0.0

    ideal_slope = np.sum(y) / np.sum(x)
    y_pred = ideal_slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        return 0.0, ideal_slope

    return float(1 - ss_res / ss_tot), float(ideal_slope)


def calculate_fractional_composition_std(polymer_array: list, monomer_id: int):
    """
    Returns (std_dev, mean_fraction) of the per-chain monomer fraction for
    the given monomer_id.

    A std_dev of 0.0 means the fraction is perfectly constant across all chains
    regardless of chain length (PDI-independent metric).

    Previously named calculate_fractional_fit_r2 but the name and docstring
    were misleading — the function never returned an R² value.
    """
    fractions = []
    for chain in polymer_array:
        dp = len(chain)
        if dp > 0:
            count = _chain_counter(chain)[monomer_id]
            fractions.append(count / dp)

    if len(fractions) < 2:
        return 0.0, 0.0

    frac_arr = np.array(fractions)
    return float(np.std(frac_arr)), float(np.mean(frac_arr))


def calculate_compositional_fit(polymer_array: list) -> dict:
    """
    Returns both absolute (PDI-sensitive) and fractional (PDI-independent)
    composition metrics.
    """
    results = {
        'pairwise_m1_m2': {},
        'vs_dp': {},
    }
    if not polymer_array:
        return results

    # Pre-compute per-chain counters once — avoids O(n²) repeated .count() calls
    chain_counters = [_chain_counter(chain) for chain in polymer_array]
    all_monomer_ids = set(mid for c in chain_counters for mid in c)
    if not all_monomer_ids:
        return results
    max_mon = max(all_monomer_ids)

    # 1. Absolute pairwise: M1 count vs M2 count
    try:
        c1 = np.array([c[1] for c in chain_counters])
        c2 = np.array([c[2] for c in chain_counters])
        if len(c1) > 1:
            r2, slope = calculate_ideal_fit_r2(c2, c1)
            results['pairwise_m1_m2'] = {'r2': r2, 'slope': slope}
    except Exception:
        logger.warning("calculate_compositional_fit: pairwise M1/M2 step failed.", exc_info=True)

    # 2. Absolute count vs DP + fractional std per monomer
    try:
        dps = np.array([len(chain) for chain in polymer_array])
        for mid in range(1, max_mon + 1):
            counts = np.array([c[mid] for c in chain_counters])
            if len(dps) > 1:
                r2, slope = calculate_ideal_fit_r2(dps, counts)
                frac_std, mean_frac = calculate_fractional_composition_std(polymer_array, mid)
                results['vs_dp'][mid] = {
                    'r2_absolute': r2,
                    'slope': slope,
                    'fractional_std': frac_std,
                    'mean_fraction': mean_frac,
                }
    except Exception:
        logger.warning("calculate_compositional_fit: vs_dp step failed.", exc_info=True)

    return results


# ---------------------------------------------------------------------------
# Basic composition statistics
# ---------------------------------------------------------------------------

def calculate_average_composition(polymer_array: list) -> dict:
    """Returns {monomer_id: percentage} for the whole ensemble."""
    if not polymer_array:
        return {}
    all_monomers = [m for p in polymer_array for m in p]
    if not all_monomers:
        return {}
    total = len(all_monomers)
    counts = Counter(all_monomers)
    return {m_id: (counts[m_id] / total) * 100.0 for m_id in sorted(counts)}


# ---------------------------------------------------------------------------
# Positional / local composition data
# ---------------------------------------------------------------------------

def get_local_composition_data(polymer_array: list, num_bins: int = 100) -> pd.DataFrame:
    """Local composition fraction per normalised-position bin."""
    df = _build_positional_records(polymer_array)
    if df.empty:
        return pd.DataFrame()

    num_monomers = int(df['monomer_id'].max())
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    df['bin'] = pd.cut(df['normalized_position'], bins=bins,
                       labels=bin_centers, include_lowest=True)
    composition_counts = df.groupby(['bin', 'monomer_id']).size().unstack(fill_value=0)
    total_in_bin = composition_counts.sum(axis=1)
    local_composition = composition_counts.div(total_in_bin, axis=0).fillna(0)

    for i in range(1, num_monomers + 1):
        if i not in local_composition.columns:
            local_composition[i] = 0

    local_composition.columns = [f"Monomer {col}" for col in local_composition.columns]
    local_composition = local_composition.reindex(
        sorted(local_composition.columns, key=lambda x: int(x.split()[-1])), axis=1
    )
    local_composition.index.name = "Normalized Position"
    return local_composition


def calculate_cmm(polymer_array: list) -> dict:
    """Centre-of-mass metric: mean normalised position per monomer."""
    pos_df = get_normalized_positional_data(polymer_array)
    if pos_df.empty:
        return {}
    return pos_df.groupby('monomer_id')['normalized_position'].mean().to_dict()


def calculate_gradient_index(polymer_array: list, num_bins: int = 100) -> dict:
    """Linear regression slope and R² of composition-vs-position per monomer."""
    comp_df = get_local_composition_data(polymer_array, num_bins=num_bins)
    if comp_df.empty:
        return {}

    x_vals = comp_df.index.to_numpy()
    gradient_results = {}
    for col_name in comp_df.columns:
        monomer_id = int(col_name.split()[-1])
        y_vals = comp_df[col_name].to_numpy()
        if len(x_vals) > 1 and np.any(y_vals):
            slope, _, r_value, _, _ = linregress(x_vals, y_vals)
            gradient_results[monomer_id] = (float(slope), float(r_value ** 2))
        else:
            gradient_results[monomer_id] = (0.0, 0.0)
    return gradient_results


def get_normalized_positional_histogram_data(
        polymer_array: list,
        num_bins: int = 20,
        return_fractions: bool = False):
    """Histogram of per-monomer positional density, optionally with global fractions."""
    df = _build_positional_records(polymer_array)
    if df.empty:
        return (pd.DataFrame(), {}) if return_fractions else pd.DataFrame()

    num_monomers = int(df['monomer_id'].max())
    total_monomer_count = len(df)
    monomer_fractions = {
        i: len(df[df['monomer_id'] == i]) / total_monomer_count
        for i in range(1, num_monomers + 1)
    }

    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    all_monomer_hists = {}

    for i in range(1, num_monomers + 1):
        monomer_positions = df[df['monomer_id'] == i]['normalized_position']
        counts, _ = np.histogram(monomer_positions, bins=bins)
        total_count = counts.sum()
        all_monomer_hists[f'Monomer {i}'] = (
            counts / total_count if total_count > 0
            else np.zeros_like(counts, dtype=float)
        )

    hist_df = pd.DataFrame(all_monomer_hists, index=bin_centers)
    hist_df.index.name = "Normalized Position Bin"
    return (hist_df, monomer_fractions) if return_fractions else hist_df


def get_normalized_positional_data(polymer_array: list) -> pd.DataFrame:
    """Flat DataFrame of (monomer_id, normalized_position) for all chains."""
    return _build_positional_records(polymer_array)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_csv_data(filepath: str) -> list:
    """
    Loads polymer sequences from a CSV file.

    Each row is treated as one chain; values are comma-separated monomer IDs
    (integers, 1-indexed).  Blank rows and provenance-header comment rows (whose
    first cell begins with ``#``) are skipped, so files carrying the metadata
    header written by ``recipe_io.write_chains_csv`` load transparently.

    Args:
        filepath: Path to the CSV file.

    Returns:
        List of chains, where each chain is a list of int monomer IDs.
    """
    return recipe_io.read_chains_csv(filepath)


# ---------------------------------------------------------------------------
# Sequence statistics
# ---------------------------------------------------------------------------

def calculate_kappa_metrics(polymer_array: list) -> dict:
    """
    Chance-corrected neighbour agreement (docs/DESCRIPTORS.md).

    P_same  fraction of neighbouring pairs whose two units are identical
    chance  sum_i f_i^2: P_same of a random chain with the same composition
    P_min   lowest P_same the composition allows: 0 if no monomer exceeds half
            the units, otherwise 2 f_max - 1
    kappa   (P_same - chance) / (1 - chance)
    kappa*  kappa for P_same >= chance; (P_same - chance) / (chance - P_min)
            below it. +1 all neighbours identical, 0 random, -1 as alternating
            as the composition allows, for any number of monomers. For an
            equimolar two-monomer chain kappa* = kappa = Guzman's lambda_2.

    Counted over all chains together. A homopolymer (chance = 1) is reported
    as kappa = kappa* = +1 (every neighbour identical).
    """
    same = pairs = 0
    counts: Counter = Counter()
    for chain in polymer_array:
        counts.update(chain)
        for a, b in zip(chain, chain[1:]):
            same += a == b
            pairs += 1
    total = sum(counts.values())
    if pairs == 0 or total == 0:
        return {'kappa': 0.0, 'kappa_star': 0.0, 'p_same': 0.0, 'chance': 0.0, 'p_min': 0.0}
    f = np.array([c / total for c in counts.values()])
    chance = float((f ** 2).sum())
    p_same = same / pairs
    p_min = max(0.0, 2.0 * float(f.max()) - 1.0)
    if chance >= 1.0 - 1e-12:
        kappa = kappa_star = 1.0
    else:
        kappa = (p_same - chance) / (1.0 - chance)
        if p_same >= chance:
            kappa_star = kappa
        else:
            kappa_star = (p_same - chance) / (chance - p_min) if chance > p_min else -1.0
    return {'kappa': float(kappa), 'kappa_star': float(kappa_star), 'p_same': float(p_same),
            'chance': chance, 'p_min': p_min}


def calculate_uncertainty_coefficient(polymer_array: list) -> float:
    """
    Order U: Theil's uncertainty coefficient of the next unit given the current
    one, over all neighbouring pairs of the ensemble:

        U = [H(next) - H(next|current)] / H(next) = 1 - H(next|current) / H(next)

    H(next|current) = H(pairs) - H(current units of the pairs) (chain rule) and
    H(next) is the entropy of the second units of the pairs. U = 0 for a random
    chain of any composition (the current unit tells nothing about the next),
    1 when every unit determines the next. A homopolymer is reported as U = 1.
    See docs/DESCRIPTORS.md.
    """
    pair_counts: Counter = Counter()
    current: Counter = Counter()
    nxt: Counter = Counter()
    for chain in polymer_array:
        for a, b in zip(chain, chain[1:]):
            pair_counts[(a, b)] += 1
            current[a] += 1
            nxt[b] += 1

    def entropy(counter):
        tot = sum(counter.values())
        return -sum((c / tot) * np.log2(c / tot) for c in counter.values() if c) if tot else 0.0

    h_next = entropy(nxt)
    if h_next <= 1e-12 or not pair_counts:
        return 1.0
    conditional = entropy(pair_counts) - entropy(current)
    return float(min(1.0, max(0.0, 1.0 - conditional / h_next)))


def calculate_mean_run_length(polymer_array: list) -> float:
    """Mean length of consecutive same-monomer runs across all chains."""
    if not polymer_array:
        return 0.0
    all_run_lengths = []
    for chain in polymer_array:
        if not chain:
            continue
        current_run = 1
        for i in range(1, len(chain)):
            if chain[i] == chain[i - 1]:
                current_run += 1
            else:
                all_run_lengths.append(current_run)
                current_run = 1
        all_run_lengths.append(current_run)
    return float(np.mean(all_run_lengths)) if all_run_lengths else 0.0


def calculate_pdi(polymer_array: list) -> dict:
    """Number-average DP (Mn), weight-average DP (Mw), and dispersity (Đ)."""
    if not polymer_array or len(polymer_array) < 2:
        return {"Mn_DP": 0, "Mw_DP": 0, "PDI": 0}
    dps = np.array([len(p) for p in polymer_array])
    mn_dp = float(np.mean(dps))
    mw_dp = float(np.sum(dps ** 2) / np.sum(dps))
    pdi = mw_dp / mn_dp if mn_dp > 0 else 0.0
    return {"Mn_DP": mn_dp, "Mw_DP": mw_dp, "PDI": pdi}


def calculate_monomer_occurrence(initial_amounts: list,
                                 amounts_history: np.ndarray) -> np.ndarray:
    """
    Fraction of each monomer in each consumption step.

    initial_amounts is used only when amounts_history does NOT already contain
    the t=0 row (legacy callers).  Since simulation_engine now captures t=0 as
    the first row, callers that pass the engine's history directly will get a
    correct result without prepending.
    """
    if amounts_history is None or amounts_history.size == 0:
        return np.array([])
    initial_amounts_np = np.array(initial_amounts).reshape(1, -1)
    full_amounts_hist = np.vstack([initial_amounts_np, amounts_history])
    consumed_in_steps = -np.diff(full_amounts_hist, axis=0)
    total_consumed_per_step = np.sum(consumed_in_steps, axis=1, keepdims=True)
    occurrence_history = np.divide(
        consumed_in_steps, total_consumed_per_step,
        out=np.zeros_like(consumed_in_steps, dtype=float),
        where=total_consumed_per_step != 0,
    )
    return occurrence_history


def calculate_run_lengths(polymer_array: list, monomer_id: int):
    """Run-length distribution for one monomer across all chains."""
    run_lengths = []
    for chain in polymer_array:
        current_run = 0
        for monomer in chain:
            if monomer == monomer_id:
                current_run += 1
            else:
                if current_run > 0:
                    run_lengths.append(current_run)
                current_run = 0
        if current_run > 0:
            run_lengths.append(current_run)
    if not run_lengths:
        return [], []
    counts = Counter(run_lengths)
    lengths = np.array(list(counts.keys()))
    counts_arr = np.array(list(counts.values()))
    return lengths, counts_arr


def calculate_all_run_lengths(polymer_array: list, num_monomers: int) -> dict:
    """Run-length distributions for all monomers."""
    all_data = {}
    for i in range(1, num_monomers + 1):
        lengths, counts = calculate_run_lengths(polymer_array, i)
        if len(lengths) > 0:
            all_data[i] = (lengths, counts)
    return all_data


def get_composition_data(polymer_array: list, num_monomers: int) -> pd.DataFrame:
    """Per-chain monomer fraction DataFrame, shape (n_chains, n_monomers)."""
    compositions = []
    for chain in polymer_array:
        chain_len = len(chain)
        if chain_len > 0:
            counts = _chain_counter(chain)
            compositions.append([counts.get(i + 1, 0) / chain_len
                                  for i in range(num_monomers)])
    return pd.DataFrame(
        compositions,
        columns=[f"Monomer {i + 1}" for i in range(num_monomers)]
    )


def calculate_multivariate_spread(polymer_array: list) -> dict:
    """
    Total multivariate compositional variance: sum of per-monomer fraction variances.
    0.0 = every chain has identical composition. Higher = more heterogeneous.
    Uses Counter-based fraction calculation (O(n) per chain, not O(n × m)).
    """
    if not polymer_array:
        return {'total_variance': 0.0, 'per_monomer_variance': np.array([]), 'mean_composition': np.array([])}

    all_mons = [m for p in polymer_array for m in p]
    n_mons = max(all_mons) if all_mons else 0
    n_chains = len(polymer_array)

    fractions = np.zeros((n_chains, n_mons))
    for i, chain in enumerate(polymer_array):
        length = len(chain)
        if length == 0:
            continue
        counts = _chain_counter(chain)
        for mid in range(1, n_mons + 1):
            fractions[i, mid - 1] = counts[mid] / length

    variances = np.var(fractions, axis=0)
    return {
        'total_variance': float(np.sum(variances)),
        'per_monomer_variance': variances,
        'mean_composition': np.mean(fractions, axis=0),
    }
