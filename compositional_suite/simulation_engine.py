"""
simulation_engine.py
--------------------
Stochastic copolymerisation simulation engine.

Key entry points
~~~~~~~~~~~~~~~~
* ``get_monomer_amounts``          – convert monomer ratios to integer pool sizes
* ``get_rate_constants_from_r_simplified`` – build k[i][j] matrix from reactivity ratios
* ``weighted_choice``              – stochastic monomer selection
* ``SimulationEngine``             – stateful engine that runs one or more polymerisation blocks

Design notes
~~~~~~~~~~~~
- Rate constants live on SimulationEngine, not on individual Polymer objects,
  to avoid redundant per-chain storage.
- Propagation history captures t=0 as the first row so that consumption curves
  have no off-by-one gap at the start.
- Degenerative (RAFT) chain transfer is modelled via a geometric burst-length
  draw parameterised by the chain transfer constant ``Ctr = k_tr / k_p``.
  ``Ctr = IDEAL_EXCHANGE`` (infinity) degenerates to sequential growth
  (one monomer per activation), the narrowest-dispersity limit.

Chain transfer model
~~~~~~~~~~~~~~~~~~~~
A propagating radical either adds monomer (rate ``k_p[M]``) or undergoes
degenerative exchange with the RAFT agent (rate ``k_tr[CTA]``).  The number of
monomers added per activation event is therefore geometric (support 1, 2, ...)
with mean ``1 + k_p[M] / (k_tr[CTA]) = 1 + ([M]/[CTA]) / Ctr``.

``Ctr`` is indexed by *terminal monomer*.  Both ``k_tr`` and ``k_p`` depend on
which unit carries the radical — a styryl radical and an acrylate radical
neither propagate nor exchange at the same rate — so in a copolymerisation the
transfer constant is a vector, exactly as ``k_p`` is already a matrix here
rather than a single constant.  A scalar is accepted and broadcast to every
monomer, which is the right approximation only for chemically similar
comonomers.

``N_CHAINs`` *is* the CTA count: the monomer pool for a block is seeded as
``N_CHAINs * BLOCK_LEN``, so ``[M]0/[CTA]0 == BLOCK_LEN`` by construction and
``sum(monomerAmounts) / N_CHAINs`` is the instantaneous ``[M]/[CTA]`` ratio.

Caveat: this engine has no radical generation, no termination, and all chains
exist from t=0.  ``Ctr`` therefore captures dispersity broadening from *slow
exchange only* — not dead chains, slow RAFT-agent consumption (hybrid
behaviour), or retardation.  It is an effective exchange constant in an
idealised living system.
"""

import random
import math
import logging
import numpy as np
from .polymer import Polymer
from collections import Counter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Version of the simulation engine. Stamped into exported dataset metadata so
# that results can always be traced back to the engine that produced them.
# Bump this whenever propagation/transfer behaviour changes in a way that would
# alter simulation output.
# 2.0.0 – CTP (unitless fudge factor) replaced by the physical chain transfer
#         constant Ctr = k_tr/k_p.
# 2.1.0 – a monomer with two or more r_ij = 0 entries chooses among those
#         comonomers in proportion to concentration (was: uniformly). Output
#         is unchanged for every other input, including all seeded binary runs.
ENGINE_VERSION = "2.1.0"

# Ctr -> infinity: exchange is instantaneous relative to propagation, so every
# activation adds exactly one monomer. The narrowest-dispersity (Poisson) limit.
IDEAL_EXCHANGE = math.inf

_MAX_RETRY_MULTIPLIER = 2   # max_retries = N_CHAINs * this
_HISTORY_SAMPLE_INTERVAL = 10  # record monomer amounts every N consumed


# ---------------------------------------------------------------------------
# Analytical dispersity relations (degenerative transfer)
# ---------------------------------------------------------------------------
# Müller/Fukuda expression for degenerative chain transfer:
#
#     Đ = 1 + 1/DPn + (2 - x) / (x * Ctr)
#
# where x is fractional conversion and DPn the number-average DP reached at
# that conversion. These closed forms let the GUI/CLI invert a target Đ
# analytically instead of bisecting over repeated simulations.

def normalise_ctr(ctr, n_monomers: int) -> list:
    """
    Coerces a chain transfer constant into a per-monomer list of length
    ``n_monomers``, validating every entry.

    Accepts a scalar (broadcast to all monomers) or a sequence with one entry
    per monomer type, indexed by terminal monomer id (1-indexed -> position 0).
    """
    if isinstance(ctr, (list, tuple)):
        values = [float(c) for c in ctr]
        if len(values) != n_monomers:
            raise ValueError(
                f"Ctr must have one entry per monomer ({n_monomers}), "
                f"got {len(values)}."
            )
    else:
        values = [float(ctr)] * n_monomers

    for i, c in enumerate(values):
        if math.isnan(c) or c <= 0:
            raise ValueError(
                f"Ctr[{i}] (chain transfer constant) must be > 0, got {c!r}."
            )
    return values


def effective_ctr(ctr, composition=None) -> float:
    """
    Collapses a per-monomer Ctr vector to the single effective constant that
    governs overall dispersity.

    Mean burst length averaged over activations is
    ``([M]/[CTA]) * sum_i F_i / Ctr_i``, where ``F_i`` is the fraction of
    activations whose terminal unit is monomer i (i.e. the copolymer
    composition). The effective constant is therefore the composition-weighted
    *harmonic* mean, not the arithmetic one — a single slow-exchanging monomer
    dominates.

    IMPORTANT — this matches the mean burst length exactly, but it is only an
    approximation for Đ. Mixing two burst distributions adds variance that a
    single mean-matched constant cannot represent, so feeding this into
    ``dispersity_from_ctr`` *underestimates* the true dispersity, increasingly
    so as the entries spread apart. Measured against simulation at DP=100,
    X=50%, r=1:

        max/min ratio    1      2      5     20    100
        Đ error       +0.03  -0.02  -0.06  -0.09  -0.23

    Treat it as reliable only when the entries lie within roughly 4x of each
    other; beyond that, simulate rather than trusting the closed form. This is
    why target-Đ solving is restricted to a uniform (scalar) Ctr.

    Args:
        ctr:         scalar or per-monomer sequence.
        composition: weights per monomer (need not be normalised). Defaults to
                     an equal split.
    """
    values = list(ctr) if isinstance(ctr, (list, tuple)) else [float(ctr)]
    if len(values) == 1:
        return values[0]

    if composition is None:
        weights = [1.0] * len(values)
    else:
        weights = [float(w) for w in composition]
        if len(weights) != len(values):
            raise ValueError("composition must match the length of ctr.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("composition weights must sum to a positive value.")

    inv = sum((w / total) / c for w, c in zip(weights, values))
    return math.inf if inv <= 0 else 1.0 / inv


def dpn_at_conversion(block_len: float, conversion_pct: float) -> float:
    """
    Number-average DP actually reached at ``conversion_pct``.

    The pool is seeded at ``N_CHAINs * block_len`` but only ``x`` of it is
    consumed, so DPn is ``x * block_len`` — not ``block_len`` itself.
    """
    return float(block_len) * (float(conversion_pct) / 100.0)


def dispersity_floor(block_len: float, conversion_pct: float) -> float:
    """
    Narrowest achievable Đ at these conditions, i.e. the ``Ctr -> inf`` limit
    ``1 + 1/DPn``. No chain transfer constant, however large, beats this.
    """
    dpn = dpn_at_conversion(block_len, conversion_pct)
    if dpn <= 0:
        return math.inf
    return 1.0 + 1.0 / dpn


def dispersity_from_ctr(ctr: float, block_len: float,
                        conversion_pct: float) -> float:
    """Predicted Đ for a given chain transfer constant."""
    floor = dispersity_floor(block_len, conversion_pct)
    if not math.isfinite(ctr):
        return floor
    x = float(conversion_pct) / 100.0
    if x <= 0 or ctr <= 0:
        return math.inf
    return floor + (2.0 - x) / (x * ctr)


def ctr_for_target_dispersity(target_pdi: float, block_len: float,
                              conversion_pct: float):
    """
    Inverts the dispersity relation for ``Ctr``.

    Returns:
        The required chain transfer constant, or ``None`` if ``target_pdi`` is
        at or below the ``1 + 1/DPn`` floor (unreachable at any ``Ctr``).
    """
    floor = dispersity_floor(block_len, conversion_pct)
    excess = float(target_pdi) - floor
    if excess <= 0:
        return None
    x = float(conversion_pct) / 100.0
    if x <= 0:
        return None
    return (2.0 - x) / (x * excess)


# Indicative bands for interpreting a Ctr value against reported RAFT
# chemistry. Ctr depends on the RAFT-agent/monomer *pair*, so these are
# order-of-magnitude guides for user feedback, not lookup values.
_CTR_BANDS = (
    (1.0,    "negligible control",
     "Below ~1 the RAFT agent barely competes with propagation; this is "
     "essentially conventional free-radical behaviour, not controlled "
     "polymerisation."),
    (10.0,   "weak control",
     "Typical of poorly matched pairs (e.g. xanthates/dithiocarbamates with "
     "activated monomers). Expect broad, drifting distributions."),
    (100.0,  "moderate control",
     "A plausible regime for well-matched trithiocarbonates."),
    (1000.0, "good control",
     "Typical of well-matched dithiobenzoates with styrenics/acrylates."),
    (1e4,    "very strong control",
     "At the top of the range reported for real RAFT agents."),
)


def classify_ctr(ctr: float):
    """
    Maps a chain transfer constant onto an indicative chemical regime.

    Returns:
        ``(label, note, plausible)`` where ``plausible`` is False when the
        value sits beyond what real RAFT agents are reported to achieve.
    """
    if not math.isfinite(ctr):
        return ("ideal exchange", "Instantaneous exchange — an idealisation, "
                                  "not attainable by a real RAFT agent.", False)
    for upper, label, note in _CTR_BANDS:
        if ctr < upper:
            return (label, note, ctr >= 1.0)
    return ("beyond reported chemistry",
            "Exceeds the exchange constants reported for known RAFT agents. "
            "The simulation will run, but no real reagent is likely to "
            "reproduce it.", False)


def get_monomer_amounts(MRs: list, N_MONs: int, MP: int) -> list:
    """
    Converts monomer feed ratios to integer pool sizes that sum to approximately MP.

    Args:
        MRs:    List of monomer ratios (any non-negative floats; need not sum to 1).
        N_MONs: Number of monomer types.
        MP:     Target total monomer pool size (integer).

    Returns:
        List of integer monomer amounts, one per monomer type.
        All zeros if MRs sums to zero.
    """
    monomerAmounts = []
    sumMonomerRatios = sum(MRs)
    if sumMonomerRatios == 0:
        return [0] * N_MONs
    for i in range(N_MONs):
        monomerAmounts.append(int(MRs[i] / sumMonomerRatios * MP))
    return monomerAmounts


def check_r_values(R_values: list) -> list:
    """
    Returns human-readable warnings about a reactivity-ratio matrix.

    The engine encodes ``r_ij = 0`` as "chain end i never homopropagates": it
    sets ``k_ii = 0`` and ``k_ij = inf``, so a chain ending in i *always* adds
    monomer j while any is left. In a binary system that is exactly what
    r = 0 means. In a row with three or more entries it also silently
    overrides every other ``r_ik`` in that row, which is rarely what was
    intended when a published value merely rounds to zero.
    """
    warnings = []
    n = len(R_values)
    for i, row in enumerate(R_values):
        others = [j for j in range(n) if j != i]
        for pos, r in enumerate(row):
            if r is None:
                warnings.append(f"r_{i + 1}{others[pos] + 1} is not set.")
            elif r < 0:
                warnings.append(
                    f"r_{i + 1}{others[pos] + 1} = {r} is negative; reactivity ratios "
                    "must be >= 0 (negative published values are fitting artefacts).")
        zeros = [pos for pos, r in enumerate(row) if r == 0]
        if zeros and len(row) > 1 and len(zeros) < len(row):
            j = others[zeros[0]] + 1
            ignored = ", ".join(f"r_{i + 1}{others[p] + 1}" for p in range(len(row))
                                if p not in zeros)
            warnings.append(
                f"r_{i + 1}{j} = 0: chains ending in M{i + 1} will always add M{j} "
                f"while any is left, and {ignored} have no effect. Use a small "
                "positive value (e.g. 0.001) if you meant 'strongly prefers "
                f"cross-propagation to M{j}'.")
    return warnings


def get_rate_constants_from_r_simplified(N_MONs: int, R_cross_values: list) -> list:
    """
    Converts cross-propagation reactivity ratios (r_ij) to a rate constant matrix.

    Assumes k_ii = 1.0 by default. If any r_ij for a given chain-end i is 0,
    homopropagation is explicitly forbidden by setting k_ii = 0.

    Bug fix: the original used a fragile shared r_idx counter that relied on
    loop ordering to stay aligned. We now enumerate the cross pairs explicitly.
    """
    rateConstantList = np.zeros((N_MONs, N_MONs))

    for i in range(N_MONs):
        k_ii = 0.0 if 0 in R_cross_values[i] else 1.0
        rateConstantList[i, i] = k_ii

        # Iterate over cross pairs in the same order the caller builds R_cross_values.
        # r_idx is a simple 0-based counter over the off-diagonal entries for row i.
        r_idx = 0
        for j in range(N_MONs):
            if j == i:
                continue
            r_ij = R_cross_values[i][r_idx]
            rateConstantList[i, j] = (1.0 / r_ij) if r_ij != 0 else math.inf
            r_idx += 1

    return rateConstantList.tolist()


def weighted_choice(choices: list):
    """
    Picks an item from [(item, weight), ...] proportional to weight.

    Infinite-weight items are always chosen (forced selection).
    The final fallback to the last item is intentional — it handles the rare
    floating-point rounding case where the running sum narrowly undershoots
    `total`. This is documented here to make it clear it is not a bug.
    """
    valid_choices = [(item, w) for item, w in choices if w > 0]
    if not valid_choices:
        raise ValueError("No valid monomers available for this chain end.")

    inf_choices = [item for item, w in valid_choices if not np.isfinite(w)]
    if inf_choices:
        return random.choice(inf_choices)

    total = sum(w for _, w in valid_choices)
    r = random.uniform(0, total)
    upto = 0.0
    for item, w in valid_choices:
        if upto + w >= r:
            return item
        upto += w

    return valid_choices[-1][0]


class SimulationEngine:
    """
    Stateful copolymerisation engine.

    A single engine instance manages one growing ensemble of polymer chains
    across potentially multiple sequential monomer-feed blocks.

    Attributes:
        N_MONs (int):              Number of distinct monomer types.
        N_CHAINs (int):            Number of chains in the ensemble.
        R_values (list):           Cross-propagation reactivity ratios, as provided by the caller.
        rateConstantList (list):   2-D rate constant matrix k[i][j] (1-indexed monomers).
        Ctr (list):                Chain transfer constants k_tr/k_p (> 0), one
                                   per terminal monomer; ``Ctr[i-1]`` applies
                                   when monomer ``i`` carries the radical. A
                                   scalar input is broadcast to all monomers.
                                   Larger = faster exchange = narrower Đ.
                                   ``IDEAL_EXCHANGE`` gives one monomer per step.
                                   Fixed for the whole synthesis: you do not
                                   swap RAFT agents between blocks.
        polymerArray (list):       Growing list of Polymer objects.
        monomerAmounts (list):     Current integer pool sizes, updated during propagation.
        block_num (int):           1-based counter of the current block.
        block_len (int):           Target average DP for the current block.
    """

    def __init__(self, params: dict):
        """
        Args:
            params: Configuration dictionary with keys:
                - 'N_MONs'   (int)             – number of monomer types
                - 'N_CHAINs' (int)             – number of chains to simulate
                - 'R_values' (list)            – cross-propagation reactivity ratios
                - 'Ctr'      (float | list)    – chain transfer constant k_tr/k_p,
                                                 scalar or one entry per monomer
                - 'seed'     (int, optional)   – if given, seeds ``random`` and
                                                 ``numpy.random`` so the entire
                                                 propagation is reproducible.
                                                 ``None`` / absent leaves the
                                                 RNGs in their current state.
        """
        seed = params.get('seed')
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed))
        self.seed = seed

        self.N_MONs: int = params['N_MONs']
        self.N_CHAINs: int = params['N_CHAINs']
        self.R_values: list = params['R_values']
        self.rateConstantList: list = get_rate_constants_from_r_simplified(
            self.N_MONs, self.R_values
        )
        # Ctr is stored per terminal monomer (1-indexed monomer id -> Ctr[id-1]).
        # A scalar broadcasts to every monomer.
        self.Ctr: list = normalise_ctr(params['Ctr'], self.N_MONs)

        self.polymerArray: list = []
        self.monomerAmounts: list = [0] * self.N_MONs
        self.initial_monomer_amounts_in_block: list = []

        # block_len is set when a block starts and is used by run_propagation.
        # Storing it on self removes the need to pass it as a parameter all the
        # way down through the call stack.
        self.block_len: int = 1
        self.block_num: int = 0
        # One record per completed block, exactly as run. Exports and saved
        # recipes are built from this rather than from GUI widgets, so a
        # dataset always describes what was actually simulated.
        self.block_history: list = []

    def _record_block(self, MRs, BLOCK_LEN, conversion, purge, result):
        self.block_history.append({
            "mrs": [float(m) for m in MRs],
            "dp": int(BLOCK_LEN),
            "conversion": float(conversion),
            "purge": bool(purge),
            "status": result[2],
        })
        return result

    # ------------------------------------------------------------------
    # Rate constant helpers
    # ------------------------------------------------------------------
    def rate_constant(self, chain_end_monomer: int, next_monomer: int) -> float:
        """
        Returns the rate constant k[chain_end][next_monomer].
        Both arguments are 1-indexed monomer IDs.
        """
        return self.rateConstantList[chain_end_monomer - 1][next_monomer - 1]

    def _can_chain_propagate(self, polymer: Polymer) -> bool:
        last = polymer.lastMonomer()
        for monomer_id in range(1, self.N_MONs + 1):
            if (self.monomerAmounts[monomer_id - 1] > 0
                    and self.rate_constant(last, monomer_id) > 0):
                return True
        return False

    def _check_feed(self, MRs):
        if len(MRs) != self.N_MONs:
            raise ValueError(
                f"Feed has {len(MRs)} monomer ratios but the engine was built for "
                f"{self.N_MONs} monomers.")
        if any(m < 0 for m in MRs) or sum(MRs) <= 0:
            raise ValueError("Monomer ratios must be >= 0 and not all zero.")

    # ------------------------------------------------------------------
    # Public simulation interface
    # ------------------------------------------------------------------
    def initialize_chains(self, MRs: list, BLOCK_LEN: int, conversion: float):
        """
        Creates the chain ensemble and runs the first propagation block.

        Each chain is seeded with a single starting monomer drawn proportional
        to the initial monomer pool, then propagation continues until
        ``conversion`` % of the pool is consumed.

        Args:
            MRs:        Monomer feed ratios for this block.
            BLOCK_LEN:  Target average degree of polymerisation for this block.
            conversion: Target monomer conversion (0–100 %).

        Returns:
            Tuple of (conversion_history, monomerAmounts_history, status).
            See ``run_propagation`` for details.
        """
        self._check_feed(MRs)
        self.block_num = 1
        self.block_len = BLOCK_LEN
        MP = int(self.N_CHAINs * BLOCK_LEN + 1)
        self.monomerAmounts = get_monomer_amounts(MRs, self.N_MONs, MP)

        for _ in range(self.N_CHAINs):
            choices = [[i + 1, self.monomerAmounts[i]] for i in range(self.N_MONs)]
            starting_monomer = weighted_choice(choices)
            polymer = Polymer([starting_monomer])
            self.polymerArray.append(polymer)
            self.monomerAmounts[starting_monomer - 1] -= 1

        return self._record_block(MRs, BLOCK_LEN, conversion, False,
                                  self.run_propagation(conversion))

    def add_new_block(self, MRs: list, BLOCK_LEN: int, conversion: float,
                      purge_monomers: bool = False):
        """
        Adds a new monomer feed block to the existing chain ensemble.

        Optionally purges any remaining monomers from the previous block before
        adding the new feed (useful for hard block-copolymer synthesis where
        residual monomer should not persist into the next block).

        Args:
            MRs:            Monomer feed ratios for the new block.
            BLOCK_LEN:      Target average DP increment for this block.
            conversion:     Target monomer conversion for this block (0–100 %).
            purge_monomers: If True, zeroes out remaining monomers before adding
                            the new feed. Default False.

        Returns:
            Tuple of (conversion_history, monomerAmounts_history, status).
            See ``run_propagation`` for details.
        """
        self._check_feed(MRs)
        self.block_num += 1
        self.block_len = BLOCK_LEN

        if purge_monomers:
            self.monomerAmounts = [0] * self.N_MONs

        monomers_to_add = get_monomer_amounts(
            MRs, self.N_MONs, int(self.N_CHAINs * BLOCK_LEN)
        )
        for i in range(self.N_MONs):
            self.monomerAmounts[i] += monomers_to_add[i]

        return self._record_block(MRs, BLOCK_LEN, conversion, purge_monomers,
                                  self.run_propagation(conversion))

    # ------------------------------------------------------------------
    # Core propagation loop
    # ------------------------------------------------------------------
    def run_propagation(self, conversion: float):
        """
        Propagates chains until `conversion`% of the current monomer pool
        is consumed.

        History fix: the initial monomer state (before any consumption) is
        now captured as the first row of monomerAmounts_history, so there is
        no off-by-one gap at t=0 when reconstructing consumption curves.

        Returns (conversion_history, monomerAmounts_history, status).
        status is "ok" or "stuck" (polymerization halted before target conversion).
        """
        initial_pool_size = sum(self.monomerAmounts)
        self.initial_monomer_amounts_in_block = self.monomerAmounts.copy()

        if initial_pool_size == 0:
            return ([], np.array([]), "ok")

        monomers_to_consume = int(initial_pool_size * (conversion / 100.0))

        conversion_history = [0.0]
        monomerAmounts_history = [self.monomerAmounts.copy()]

        consumed_count = 0
        status = "ok"
        max_retries = self.N_CHAINs * _MAX_RETRY_MULTIPLIER

        while consumed_count < monomers_to_consume and sum(self.monomerAmounts) > 0:
            # Select a chain that can still propagate
            polymer = None
            for _ in range(max_retries):
                candidate = random.choice(self.polymerArray)
                if self._can_chain_propagate(candidate):
                    polymer = candidate
                    break

            if polymer is None:
                status = "stuck"
                logger.warning(
                    "Propagation stuck at %d/%d monomers consumed.",
                    consumed_count, monomers_to_consume,
                )
                break

            # The burst is set by the terminal unit holding the radical, since
            # both k_tr and k_p depend on it.
            num_to_add = self._draw_burst_length(
                consumed_count, monomers_to_consume, polymer.lastMonomer()
            )

            for _ in range(num_to_add):
                if consumed_count >= monomers_to_consume:
                    break
                if sum(self.monomerAmounts) <= 0:
                    break

                last = polymer.lastMonomer()
                choices = [
                    [mid, self.monomerAmounts[mid - 1] * self.rate_constant(last, mid)]
                    for mid in range(1, self.N_MONs + 1)
                ]
                # r_ij = 0 encodes k_ij = inf. With two or more such comonomers
                # available (a monomer that cannot homopropagate, in a 3+
                # monomer system) the ratios carry no information about the
                # relative cross rates, so they are taken as equal and the
                # choice follows concentration. (Engine <= 2.0 picked uniformly,
                # ignoring the feed.) A single forced choice is left on the
                # original code path so seeded binary results are unchanged.
                forced = [c for c in choices if not math.isfinite(c[1]) and c[1] > 0]
                if len(forced) > 1:
                    choices = [[mid, self.monomerAmounts[mid - 1] if [mid, w] in forced else 0]
                               for mid, w in choices]
                try:
                    next_monomer = weighted_choice(choices)
                except ValueError:
                    break

                polymer.append(next_monomer)
                self.monomerAmounts[next_monomer - 1] -= 1
                consumed_count += 1

                if consumed_count % _HISTORY_SAMPLE_INTERVAL == 0:
                    current_consumption = initial_pool_size - sum(self.monomerAmounts)
                    conv_pct = (
                        (current_consumption / monomers_to_consume * conversion)
                        if monomers_to_consume > 0 else 0
                    )
                    conversion_history.append(min(conv_pct, 100.0))
                    monomerAmounts_history.append(self.monomerAmounts.copy())

        return (conversion_history, np.array(monomerAmounts_history), status)

    def _draw_burst_length(self, consumed_count: int, monomers_to_consume: int,
                           terminal_monomer: int) -> int:
        """
        Determines how many monomers to add to a single chain in one activation
        event, i.e. before the chain hands the radical back to the RAFT agent.

        A radical either propagates (rate k_p[M]) or exchanges (rate k_tr[CTA]),
        so the burst is geometric with mean 1 + ([M]/[CTA]) / Ctr. The ratio is
        recomputed each step, so bursts shorten as monomer is consumed — which
        is why Đ narrows with conversion.

        Args:
            terminal_monomer: 1-indexed id of the unit carrying the radical,
                              which selects the transfer constant to use.

        With Ctr = IDEAL_EXCHANGE, always returns 1.
        """
        ctr = self.Ctr[terminal_monomer - 1]
        if not math.isfinite(ctr):
            return 1

        total_monomers_left = sum(self.monomerAmounts)
        if total_monomers_left <= 0:
            return 1

        # N_CHAINs is the CTA count, so this is the instantaneous [M]/[CTA].
        monomer_to_cta_ratio = total_monomers_left / self.N_CHAINs
        dynamic_prob_add_more = (
            monomer_to_cta_ratio / (ctr + monomer_to_cta_ratio)
        )

        monomers_left_in_block = monomers_to_consume - consumed_count
        if monomers_left_in_block <= 1:
            return 1

        num_to_add = 1
        while random.random() < dynamic_prob_add_more:
            if num_to_add >= monomers_left_in_block:
                break
            num_to_add += 1

        return num_to_add