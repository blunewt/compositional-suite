"""
plotting.py
-----------
Matplotlib/Seaborn rendering functions for polymer ensemble visualisation.

All plot functions follow the same calling convention::

    plot_xxx(ax, data, ..., title=None)

The caller is responsible for creating the ``Axes`` (and any surrounding
``Figure``) and for calling ``canvas.draw()`` or ``plt.show()`` afterwards.
Functions that require more than one axes (e.g. the KDE-with-marginals plot)
accept a tuple of axes.

Colour palette
~~~~~~~~~~~~~~
A single ``MONOMER_COLORS`` list is used throughout so that each monomer
always maps to the same colour, regardless of which plot is displayed.
Legacy aliases ``MONOMER_COLOR_LIST`` and ``STARK_COLORS`` are kept for
backwards compatibility with any external imports.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter1d

sns.set_style('whitegrid')
plt.rcParams['font.family'] = ['DejaVu Sans', 'Segoe UI', 'Arial', 'sans-serif']
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
MONOMER_COLORS = [
    '#1f77b4',  # M1 — muted blue
    '#d62728',  # M2 — brick red
    '#2ca02c',  # M3 — green
    '#ff7f0e',  # M4 — orange
    '#9467bd',  # M5 — purple
    '#8c564b',  # M6 — brown
    '#e377c2',  # M7 — pink
    '#17becf',  # M8 — cyan
]

# Keep legacy names so any external imports still work
MONOMER_COLOR_LIST = MONOMER_COLORS
STARK_COLORS = MONOMER_COLORS + ['#bcbd22', '#000000', '#7f7f7f', '#003f5c']


def _mon_color(i: int) -> str:
    """Return the colour for monomer index i (0-based)."""
    return MONOMER_COLORS[i % len(MONOMER_COLORS)]


def _no_data(ax, msg='No data.'):
    ax.text(0.5, 0.5, msg, ha='center', va='center',
            transform=ax.transAxes, fontsize=11, color='#666666')


# ---------------------------------------------------------------------------
# Run-length histogram
# ---------------------------------------------------------------------------

def plot_run_length_histogram(ax, polymer_array, n_mons, title=None):
    """
    Distribution of consecutive same-monomer run lengths.

    Fix: uses explicit Counter-based bars instead of np.bincount, so sparse
    distributions (e.g. only lengths 1, 5, 12 appear) don't produce misleading
    empty-gap bars that look like absent data.
    """
    ax.clear()
    ax.set_title(title or "Monomer Run Length Distribution")

    if not polymer_array:
        _no_data(ax)
        return

    runs_dict = {i + 1: [] for i in range(n_mons)}
    for chain in polymer_array:
        if not chain:
            continue
        current_monomer = chain[0]
        current_count = 1
        for m in chain[1:]:
            if m == current_monomer:
                current_count += 1
            else:
                runs_dict[current_monomer].append(current_count)
                current_monomer = m
                current_count = 1
        runs_dict[current_monomer].append(current_count)

    has_data = False
    for i in range(n_mons):
        mid = i + 1
        data = runs_dict[mid]
        if not data:
            continue
        has_data = True
        counts = Counter(data)
        lengths = np.array(sorted(counts.keys()))
        freqs = np.array([counts[l] for l in lengths])
        ax.bar(lengths, freqs, label=f'M{mid}',
               color=_mon_color(i), alpha=0.6, width=0.7)

    if not has_data:
        _no_data(ax, 'No run data.')
        return

    ax.set_xlabel("Run Length (Consecutive Units)")
    ax.set_ylabel("Frequency (Count)")
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()


# ---------------------------------------------------------------------------
# Multivariate parallel coordinates
# ---------------------------------------------------------------------------

def plot_multivariate_parallel_coordinates(ax, polymer_array, n_mons, title=None):
    """
    Per-chain composition across all monomers with heterogeneity index.
    Uses Counter for O(n) fraction calculation instead of O(n x m) .count() calls.
    Monomer axis labels are coloured to match the monomer colour palette.
    """
    ax.clear()
    ax.set_title(title or "Multivariate Compositional Spread")

    if not polymer_array or n_mons < 2:
        _no_data(ax, "Requires at least 2 monomers.")
        return

    all_fractions = []
    for chain in polymer_array:
        L = len(chain)
        if L > 0:
            c = Counter(chain)
            all_fractions.append([c.get(m + 1, 0) / L for m in range(n_mons)])

    frac_matrix = np.array(all_fractions)
    total_variance = float(np.sum(np.var(frac_matrix, axis=0)))

    n_chains = len(frac_matrix)
    indices = np.random.choice(n_chains, min(300, n_chains), replace=False)
    x = np.arange(n_mons)

    for idx in indices:
        ax.plot(x, frac_matrix[idx], color='#888888', alpha=0.12, linewidth=0.5)

    mean_comp = np.mean(frac_matrix, axis=0)
    ax.plot(x, mean_comp, color='#222222', linewidth=2.5,
            marker='o', markersize=6, label='Mean', zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"M{i + 1}" for i in range(n_mons)])
    for tick, i in zip(ax.get_xticklabels(), range(n_mons)):
        tick.set_color(_mon_color(i))
        tick.set_fontweight('bold')

    ax.text(0.05, 0.95, f"Total Variance\n{total_variance:.4f}",
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.85, edgecolor='#333333'))

    ax.set_ylabel("Monomer Fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize='small')


# ---------------------------------------------------------------------------
# Eigenvectors
# ---------------------------------------------------------------------------

def plot_eigenvectors(ax, eigenvectors, n_mons, title=None):
    """
    Bar chart of the primary transition-matrix eigenvectors (v₂ and, for ≥ 3
    monomers, v₃).  These indicate which monomers drive the dominant
    non-stationary sequence modes.

    Args:
        ax:          Matplotlib Axes to draw on.
        eigenvectors: Sequence of up to two eigenvector arrays
                      (e.g. [v2_array, v3_array]).
        n_mons:      Number of distinct monomer types.
        title:       Optional axes title override.
    """
    ax.clear()
    ax.set_title(title or "Primary Sequence Modes (Eigenvectors)")

    if n_mons < 2:
        _no_data(ax, 'Requires at least 2 monomers.')
        return

    labels = [f'M{i + 1}' for i in range(n_mons)]
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, eigenvectors[0][:n_mons], width,
           label='v₂ (for λ₂)', color='#007ACC')
    if n_mons >= 3 and len(eigenvectors) > 1:
        ax.bar(x + width / 2, eigenvectors[1][:n_mons], width,
               label='v₃ (for λ₃)', color=_mon_color(1))

    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Component Value')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--')


# ---------------------------------------------------------------------------
# Monomer consumption
# ---------------------------------------------------------------------------

def plot_monomer_consumption(ax, conv_history, amounts_history, n_mons,
                              normalize=True, title=None):
    """
    Line chart of monomer consumption over the course of polymerisation.

    Args:
        ax:              Matplotlib Axes to draw on.
        conv_history:    List of conversion (%) values recorded during simulation.
        amounts_history: 2-D NumPy array, shape (steps, n_mons), of remaining
                         monomer pool sizes at each recorded step.
        n_mons:          Number of distinct monomer types.
        normalize:       If True (default), plot each monomer as % consumed
                         relative to its starting amount.  If False, plot
                         absolute remaining amounts.
        title:           Optional axes title override.
    """
    ax.clear()

    if not conv_history or amounts_history is None or amounts_history.size == 0:
        _no_data(ax, 'No consumption data.')
        ax.set_title(title or "Monomer Consumption")
        return

    if amounts_history.shape[0] < 2:
        _no_data(ax, 'Insufficient history data.')
        ax.set_title(title or "Monomer Consumption")
        return

    if normalize:
        initial = amounts_history[0, :]
        ax.set_title(title or "Normalised Monomer Consumption")
        for i in range(n_mons):
            if initial[i] > 0:
                consumed_pct = (initial[i] - amounts_history[:, i]) / initial[i] * 100
                ax.plot(conv_history, consumed_pct,
                        label=f"M{i + 1}", color=_mon_color(i), lw=2)
        ax.set_ylabel("Monomer Consumed (%)")
        ax.set_ylim(0, 105)
    else:
        ax.set_title(title or "Absolute Monomer Amounts")
        for i in range(n_mons):
            ax.plot(conv_history, amounts_history[:, i],
                    label=f"M{i + 1}", color=_mon_color(i), lw=2)
        ax.set_ylabel("Monomer Amount")
        ax.set_ylim(bottom=0)

    ax.set_xlabel("Conversion (%)")
    ax.set_xlim(0, 100)
    ax.legend()
    ax.grid(True)


# ---------------------------------------------------------------------------
# Sequence heatmap
# ---------------------------------------------------------------------------

def plot_sequence_heatmap(ax, polymer_array, num_seqs_to_show, n_mons, title=None):
    ax.clear()
    ax.set_title(title or 'Sequence Visualisation')

    if not polymer_array:
        _no_data(ax, 'No polymers.')
        return

    chains = polymer_array[:num_seqs_to_show]
    max_len = max(len(c) for c in chains) if chains else 0
    if max_len == 0:
        _no_data(ax, 'No polymers.')
        return

    padded_chains = [c + [0] * (max_len - len(c)) for c in chains]
    colors = ['white'] + [_mon_color(i) for i in range(n_mons)]
    custom_cmap = ListedColormap(colors)
    bounds = np.arange(n_mons + 2) - 0.5
    norm = BoundaryNorm(bounds, custom_cmap.N)

    sns.heatmap(
        padded_chains,
        cmap=custom_cmap,
        norm=norm,
        xticklabels=False,
        yticklabels=False,
        square=False,
        cbar=True,
        ax=ax,
        cbar_kws={'ticks': np.arange(1, n_mons + 1), 'label': 'Monomer ID'},
    )
    ax.set_xlabel("Position")
    ax.set_ylabel(f"First {len(chains)} Chains")


# ---------------------------------------------------------------------------
# Positional composition heatmap  (replaces plot_mirrored_kde)
#
# The mirrored KDE was misleading for 3+ monomers (runs out of "sides") and
# its normalisation simplified to smooth_y / total_system_mass — the
# smooth_total terms cancelled, making the mirroring cosmetic rather than
# informative.
#
# This 2-D heatmap shows gradient structure directly: rows = monomers,
# columns = normalised position bins, colour = fraction.  Block-like,
# gradient, and random architectures are immediately distinguishable.
# ---------------------------------------------------------------------------

def plot_positional_composition_heatmap(ax, polymer_array, n_mons,
                                         num_bins=50, title=None):
    """
    2-D heatmap: rows = monomers, columns = normalised position bins.
    Colour encodes the fraction of that monomer at that position.
    """
    ax.clear()
    ax.set_title(title or "Positional Composition Heatmap")

    if not polymer_array:
        _no_data(ax)
        return

    matrix = np.zeros((n_mons, num_bins))
    bin_totals = np.zeros(num_bins)

    for chain in polymer_array:
        chain_len = len(chain)
        if chain_len < 2:
            continue
        for pos, mid in enumerate(chain):
            norm_pos = pos / (chain_len - 1)
            b = min(int(norm_pos * num_bins), num_bins - 1)
            if 1 <= mid <= n_mons:
                matrix[mid - 1, b] += 1
                bin_totals[b] += 1

    safe_totals = np.where(bin_totals > 0, bin_totals, 1)
    matrix /= safe_totals[np.newaxis, :]

    im = ax.imshow(
        matrix,
        aspect='auto',
        origin='lower',
        cmap='RdYlGn_r',
        vmin=0, vmax=1,
        extent=[0, 1, 0.5, n_mons + 0.5],
        interpolation='nearest',
    )

    cbar = ax.get_figure().colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction", fontsize=9)

    ax.set_yticks(range(1, n_mons + 1))
    ax.set_yticklabels([f"M{i + 1}" for i in range(n_mons)], fontsize=9)
    for ytick, i in zip(ax.get_yticklabels(), range(n_mons)):
        ytick.set_color(_mon_color(i))
        ytick.set_fontweight('bold')

    ax.set_xlabel("Normalised Position")
    ax.set_ylabel("Monomer")
    ax.set_xlim(0, 1)


def plot_mirrored_kde(ax, polymer_array, n_mons, title=None):
    """
    Legacy mirrored KDE positional-density plot.

    Still used by the Simulator and Analysis tabs. For ensembles with more
    than two monomers ``plot_positional_composition_heatmap`` is more
    informative (the mirrored layout runs out of "sides").

    Odd-indexed monomers are plotted above the axis and even-indexed monomers
    below, creating a mirrored appearance.  Position is expressed as absolute
    index (not normalised).

    Args:
        ax:            Matplotlib Axes to draw on.
        polymer_array: List of chains (each chain is a list of int monomer IDs).
        n_mons:        Number of distinct monomer types.
        title:         Optional axes title override.
    """
    ax.clear()
    ax.set_title(title or "Monomer Positional Density")

    all_monomers = [monomer for chain in polymer_array for monomer in chain]
    if not all_monomers:
        ax.text(0.5, 0.5, 'No data to plot.', ha='center')
        return

    max_len = max(len(p) for p in polymer_array) if polymer_array else 0
    if max_len == 0:
        return

    counts = np.zeros((max_len, n_mons))
    for chain in polymer_array:
        for pos, mid in enumerate(chain):
            if pos < max_len:
                counts[pos, mid - 1] += 1

    sigma = max(1.5, max_len * 0.02)
    x_vals = np.arange(max_len)

    total_counts_per_pos = counts.sum(axis=1)
    smooth_total = gaussian_filter1d(total_counts_per_pos, sigma=sigma, mode='mirror')
    smooth_total = np.clip(smooth_total, 1e-9, None)

    total_system_mass = len(all_monomers)

    for i in range(n_mons):
        monomer_id = i + 1
        raw_y = counts[:, i]
        smooth_y = gaussian_filter1d(raw_y, sigma=sigma, mode='mirror')
        final_y = (smooth_y / smooth_total) * (smooth_total / total_system_mass)

        if monomer_id % 2 != 0:
            y_plot = final_y
        else:
            y_plot = -final_y

        color = _mon_color(i)
        ax.plot(x_vals, y_plot, color=color, linewidth=1.5)
        ax.fill_between(x_vals, 0, y_plot, color=color, alpha=0.5,
                        label=f'Monomer {monomer_id}')

    ax.axhline(0, color='black', lw=0.8)
    vals = ax.get_yticks()
    ax.set_yticklabels([f"{abs(x):.3f}" for x in vals])
    ax.set_xlabel("Position in Chain")
    ax.set_ylabel("Relative Density")
    ax.legend(ncol=2 if n_mons > 4 else 1, fontsize='small' if n_mons > 4 else None)
    ax.set_xlim(left=0, right=max_len)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# 2-D KDE with marginals (M1 vs M2 cone of heterogeneity)
# ---------------------------------------------------------------------------

def plot_2d_kde_with_marginals(axes, polymer_array, n_mons, title=None):
    """
    Binary cone of heterogeneity (M1 count vs M2 count).

    Defined for two-monomer copolymers only: for any other number of monomers
    the three axes are hidden and nothing is drawn.
    Uses Counter for O(n) count extraction instead of .count().
    """
    from . import analysis

    ax_main, ax_top, ax_right = axes
    ax_main.clear()
    ax_top.clear()
    ax_right.clear()
    binary = n_mons == 2
    for ax in axes:
        ax.set_visible(binary)
    if not binary:
        return

    if not polymer_array:
        _no_data(ax_main, 'No data.')
        return

    chain_counters = [Counter(p) for p in polymer_array]
    x_data = np.array([c[2] for c in chain_counters])  # M2
    y_data = np.array([c[1] for c in chain_counters])  # M1

    df = pd.DataFrame({'Monomer 1': y_data, 'Monomer 2': x_data})

    x_max = x_data.max() if len(x_data) > 0 else 1
    y_max = y_data.max() if len(y_data) > 0 else 1
    x_clip = (0, x_max * 1.05)
    y_clip = (0, y_max * 1.05)

    if np.std(x_data) > 0 and np.std(y_data) > 0:
        sns.kdeplot(data=df, x='Monomer 2', y='Monomer 1',
                    ax=ax_main, cmap='Reds', fill=True, cut=0,
                    clip=(x_clip, y_clip))
        sns.kdeplot(data=df, x='Monomer 2', ax=ax_top, fill=True,
                    color=_mon_color(1), cut=0, clip=x_clip)
        sns.kdeplot(data=df, y='Monomer 1', ax=ax_right, fill=True,
                    color=_mon_color(0), cut=0, clip=y_clip)
    else:
        ax_main.scatter(x_data, y_data, alpha=0.5, color=_mon_color(1), s=10)

    if len(x_data) > 1:
        spread_data = analysis.calculate_compositional_spread(polymer_array)
        if spread_data:
            std_f = spread_data['std_dev']
            line_x = np.linspace(0, x_clip[1], 100)

            mean_n2 = float(np.mean(x_data))
            mean_n1 = float(np.mean(y_data))
            slope_mean = mean_n1 / mean_n2 if mean_n2 > 0 else 1.0

            sum_n1n2 = y_data + x_data
            f1_binary = np.divide(y_data, sum_n1n2,
                                   out=np.zeros_like(y_data, dtype=float),
                                   where=sum_n1n2 != 0)
            mean_f = float(np.mean(f1_binary))
            std_f_binary = float(np.std(f1_binary))

            def _slope(f):
                return np.clip(f, 0.01, 0.99) / (1 - np.clip(f, 0.01, 0.99))

            ax_main.plot(line_x, slope_mean * line_x,
                         color='black', linestyle='-', linewidth=1.5, label='Avg')
            ax_main.plot(line_x, _slope(mean_f + std_f_binary) * line_x,
                         color='black', linestyle=':', alpha=0.6)
            ax_main.plot(line_x, _slope(mean_f - std_f_binary) * line_x,
                         color='black', linestyle=':', alpha=0.6)
            ax_main.fill_between(
                line_x,
                _slope(mean_f - std_f_binary) * line_x,
                _slope(mean_f + std_f_binary) * line_x,
                color='gray', alpha=0.1,
            )
            ax_main.text(0.05, 0.95,
                         f"$\\sigma_{{frac}}$\n{std_f:.4f}",
                         transform=ax_main.transAxes, fontsize=11,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white',
                                   alpha=0.9, edgecolor='#333333'))

    ax_main.set_xlim(*x_clip)
    ax_main.set_ylim(*y_clip)
    ax_main.set_xlabel("M2 Count")
    ax_main.set_ylabel("M1 Count")

    plot_title = title or "Binary cone of heterogeneity"
    if n_mons > 2:
        plot_title += f"  (M1 vs M2 only — {n_mons - 2} monomer(s) not shown)"
    ax_top.set_title(plot_title, fontsize=10)
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)
    for _ax in [ax_top, ax_right]:
        _ax.set(xlabel='', ylabel='')
        _ax.spines[['right', 'top']].set_visible(False)


# ---------------------------------------------------------------------------
# Monomer occurrence vs conversion
# ---------------------------------------------------------------------------

def plot_monomer_occurrence(ax, conv_history, occurrence_history, n_mons, title=None):
    """
    Line chart of the normalised fraction of each monomer incorporated at each
    conversion step (i.e. instantaneous monomer occurrence vs. conversion).

    Args:
        ax:                 Matplotlib Axes to draw on.
        conv_history:       List of conversion (%) values.
        occurrence_history: 2-D array of shape (steps, n_mons) containing the
                            fraction of each monomer consumed in each step
                            (output of ``analysis.calculate_monomer_occurrence``).
        n_mons:             Number of distinct monomer types.
        title:              Optional axes title override.
    """
    ax.clear()
    ax.set_title(title or "Normalised Monomer Occurrence")

    min_len = min(len(conv_history), len(occurrence_history))
    if min_len <= 1:
        _no_data(ax, 'No data to plot.')
        return

    df = pd.DataFrame(occurrence_history[:min_len],
                      columns=[f"Monomer {i + 1}" for i in range(n_mons)])
    df['Conversion'] = conv_history[:min_len]
    bins = np.linspace(0, 100, 101)
    df['Conversion Bin'] = pd.cut(df['Conversion'], bins=bins, right=False)
    binned = df.groupby('Conversion Bin', observed=True).mean(numeric_only=True)
    plot_x = [interval.mid for interval in binned.index]

    for i in range(n_mons):
        col = f"Monomer {i + 1}"
        if col in binned.columns:
            ax.plot(plot_x, binned[col], label=f"M{i + 1}",
                    color=_mon_color(i), lw=2)

    ax.set_xlabel("Conversion (%)")
    ax.set_ylabel("Normalised Monomer Occurrence")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)


# ---------------------------------------------------------------------------
# Smoothed positional distribution
# ---------------------------------------------------------------------------

def plot_smoothed_positional_distribution(ax, hist_df, monomer_fractions, title=None):
    """
    Area-under-curve plot of the Gaussian-smoothed monomer positional density
    as a function of normalised chain position.

    Args:
        ax:                Matplotlib Axes to draw on.
        hist_df:           DataFrame of per-monomer positional histograms
                           (output of ``analysis.get_normalized_positional_histogram_data``).
        monomer_fractions: Dict mapping monomer ID (int) to its global fraction
                           in the ensemble (used for density scaling).
        title:             Optional axes title override.
    """
    ax.clear()
    ax.set_title(title or "Smoothed Positional Distribution")

    if hist_df is None or hist_df.empty:
        _no_data(ax, 'No data to plot.')
        ax.set_xlim(0, 1)
        return

    x = hist_df.index.to_numpy()
    sigma = max(2, len(x) * 0.03)
    smoothed_data = {}

    for col_name in hist_df.columns:
        mid = int(col_name.split()[-1])
        fraction = monomer_fractions.get(mid, 0)
        y_smooth = gaussian_filter1d(
            hist_df[col_name].to_numpy() * fraction, sigma=sigma, mode='mirror'
        )
        smoothed_data[col_name] = y_smooth

    total_density = np.clip(sum(smoothed_data.values()), 1e-9, None)
    avg_total = float(np.mean(total_density))
    scaling = 100.0 / avg_total

    for col_name in hist_df.columns:
        mid = int(col_name.split()[-1])
        y_pct = smoothed_data[col_name] / total_density * avg_total * scaling
        color = _mon_color(mid - 1)
        ax.plot(x, y_pct, label=f"M{mid}", color=color, lw=2)
        ax.fill_between(x, 0, y_pct, alpha=0.35, color=color)

    ax.set_xlabel("Normalised Position")
    ax.set_ylabel("Relative Density (%)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True)


# ---------------------------------------------------------------------------
# Composition KDE
# ---------------------------------------------------------------------------

def plot_composition_kde(ax, comp_df, title=None):
    """
    KDE of per-chain monomer fractions — one curve per monomer.  Broad
    distributions indicate high inter-chain compositional heterogeneity.

    Args:
        ax:      Matplotlib Axes to draw on.
        comp_df: DataFrame of per-chain monomer fractions, shape
                 (n_chains, n_monomers), e.g. from ``analysis.get_composition_data``.
        title:   Optional axes title override.
    """
    ax.clear()
    ax.set_title(title or "Composition Heterogeneity")

    if comp_df is None or comp_df.empty:
        _no_data(ax, 'No composition data.')
        return

    for i, col in enumerate(comp_df.columns):
        sns.kdeplot(comp_df[col], ax=ax, label=f"M{i + 1}",
                    fill=True, alpha=0.45, color=_mon_color(i))

    ax.set_xlabel("Monomer Fraction in Chain")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_xlim(0, 1)


# ---------------------------------------------------------------------------
# DP distribution
# ---------------------------------------------------------------------------

def plot_dp_distribution(ax, polymer_array, title=None):
    """
    Number and weight fraction DP distributions as KDEs.

    Fix: fallback histogram now uses correct weight-fraction normalisation
    (weights = dps/dps.sum(), density=False) instead of the previous
    density=True with weights=dps which was incorrect.
    """
    ax.clear()
    ax.set_title(title or "DP Distribution — Number & Weight")

    if not polymer_array:
        _no_data(ax, 'No data to plot.')
        return

    dps = np.array([len(p) for p in polymer_array if p])
    if dps.size < 2:
        _no_data(ax, 'Not enough data for distribution.')
        return

    x_vals = np.linspace(0, dps.max() * 1.2, 500)

    try:
        kde_num = gaussian_kde(dps)
        y_num = kde_num(x_vals)
        ax.plot(x_vals, y_num, label='Number (Mn)', color=_mon_color(0), lw=2)
        ax.fill_between(x_vals, 0, y_num, color=_mon_color(0), alpha=0.25)

        # Correct weight-fraction: weights must sum to 1
        kde_wt = gaussian_kde(dps, weights=dps / dps.sum())
        y_wt = kde_wt(x_vals)
        ax.plot(x_vals, y_wt, label='Weight (Mw)', color=_mon_color(1), lw=2)
        ax.fill_between(x_vals, 0, y_wt, color=_mon_color(1), alpha=0.25)

    except Exception:
        bins = max(20, len(np.unique(dps)))
        ax.hist(dps, bins=bins, density=True, alpha=0.5,
                label='Number Hist', color=_mon_color(0))
        # Correct fallback weight histogram
        ax.hist(dps, bins=bins, weights=dps / dps.sum(), alpha=0.5,
                label='Weight Hist', color=_mon_color(1))

    ax.set_xlabel("Degree of Polymerisation (DP)")
    ax.set_ylabel("Normalised Intensity")
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)