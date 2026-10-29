"""
Structural deviation (SD) metrics for binary (A/B) copolymers.

Implements the framework of Conka et al. exactly as defined in

* EPJ 2023  - R. Conka et al., Eur. Polym. J. 185 (2023) 111769, and its SI
              (targets Eqs. S1-S9, per-chain SD Eqs. S10-S12, normalisation
              Eq. S14, equation-driven quality thresholds Eq. 21 / Fig. 13);
* RCE 2023  - R. Conka et al., React. Chem. Eng. 8 (2023) 2905 (multi-angle
              evaluation: <SD>, sigma_SD, skewness; f_Gr scan, Fig. 9);
* ACS Polym. Au 6 (2026) 506 (perspective; CV_SD).

SD is defined for two-monomer copolymers only. Conventions follow the papers:

* **B is the minority monomer** (B-Func = x_B <= 0.5). If the polymer's second
  monomer is the majority, the labels are swapped before scoring and the
  mapping is reported.
* A chain is compared with the target at its own length omega (Eq. S11).
* SD*(z) is the minimum over readings of the chain (Eq. S12): left->right and
  right->left (both cumulated from the chain end, as for the homopolymer in
  Eq. S14); for symmetric compositions also with the A/B labels exchanged
  (four readings). This makes the result independent of which end, or which
  label, the gradient starts from.
* SD = SD* / SD*_HP, with SD*_HP the B-homopolymer read right->left (Eq. S14).
* Quality thresholds are not fixed numbers: they come from an *ideal* sample of
  the same target, chain length and composition (EPJ 2023 Section 2.2):
  Excellent/Good = <SD_Ideal>, Good/Poor = 98th percentile of SD_Ideal. For
  block-like targets (f_Gr < 0.25) the Good/Poor value at f_Gr = 0.25 is used
  (EPJ 2023, Fig. 13c recommendation).

Published threshold values (e.g. 0.12 / 0.30 for a symmetric DP-100 gradient)
were derived from rounded inputs; computing them from the same rules as the
scored chains gives 0.10 / 0.26 for that case. This module always uses the
self-consistent values.
"""

import functools
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

SYMMETRIC_TOLERANCE = 0.02     # |x_B - 0.5| below which the 4-reading rule applies
IDEAL_SAMPLE_SIZE = 20_000     # chains in the ideal reference sample (EPJ: >=1e5 for
                               # full convergence; 2e4 moves thresholds < 0.005)
BLOCK_LIKE_FGR = 0.25          # below this, Good/Poor is evaluated at f_Gr = 0.25
STANDARD_FGR = (1.0, 0.5, 0.0) # symmetric compositions: gradient, BGB, diblock
SCAN_FGR = tuple(np.round(np.linspace(0.0, 1.0, 11), 2))


# ---------------------------------------------------------------------------
# Targets (EPJ 2023 SI, Eqs. S1-S9), B = minority monomer, x_B <= 0.5
# ---------------------------------------------------------------------------

def target_probability(omega: int, x_B: float, f_Gr: float) -> np.ndarray:
    """
    Monomer-B inclusion probability P_B(y), y = 1..omega, for a target with
    gradient fraction f_Gr (block fraction f_Bl = 1 - f_Gr).

    * f_Gr = 1           : gradient, P_B = 2 x_B (y-1)/(omega-1)        (S1, S4)
    * f_Gr = 0           : diblock, A for y <= x_A omega, then B        (S3, S9)
    * 2 x_B <= f_Gr < 1  : block(A)-gradient                            (S7)
    * 0 < f_Gr < 2 x_B   : block(A)-gradient-block(B)                   (S2, S8)

    Eq. S8 as printed in the SI gives the gradient segment the numerator
    (y - x_B omega + f omega/2); that starts at P_B = (x_A - x_B)/f instead of 0
    for asymmetric polymers. The intended form (y - (x_A - f/2) omega), which
    reduces to Eq. S2 for x_A = x_B, is used here.
    """
    if not 0.0 < x_B <= 0.5 + 1e-12:
        raise ValueError(f"x_B must be in (0, 0.5] (B is the minority monomer); got {x_B}")
    if not 0.0 <= f_Gr <= 1.0:
        raise ValueError(f"f_Gr must be in [0, 1]; got {f_Gr}")
    w = int(omega)
    y = np.arange(1, w + 1, dtype=float)
    x_A = 1.0 - x_B
    b = lambda v: round(v, 9)        # segment boundaries free of float noise (1 - 0.8 != 0.2)
    if f_Gr >= 1.0:
        p = 2.0 * x_B * (y - 1) / (w - 1) if w > 1 else np.full(w, x_B)
    elif f_Gr <= 0.0:
        p = (y > b(x_A * w)).astype(float)
    elif f_Gr >= 2.0 * x_B:
        bl = b((1.0 - f_Gr) * w)
        x_B_gr = x_B / f_Gr                                        # Eq. S5
        p = np.where(y <= bl, 0.0, 2.0 * x_B_gr * (y - bl) / (w - bl))
    else:
        start, end = b((x_A - f_Gr / 2.0) * w), b((x_A + f_Gr / 2.0) * w)
        p = np.where(y <= start, 0.0,
                     np.where(y <= end, (y - start) / (f_Gr * w), 1.0))
    return np.clip(p, 0.0, 1.0)


def structure_name(x_B: float, f_Gr: float) -> str:
    """Name of the target family member, as used in the papers."""
    if f_Gr >= 1.0:
        return "gradient"
    if f_Gr <= 0.0:
        return "diblock"
    return "block-gradient" if f_Gr >= 2.0 * x_B else "block-gradient-block"


def standard_targets(x_B: float, symmetric: bool) -> tuple:
    """
    The Conka et al. target set for a composition (B = minority, x_B <= 0.5).

    Asymmetric (x_B < 0.5): gradient (GD, f_Gr = 1), block-gradient (BGD,
    f_Gr = 2 x_B: an A block followed by a full 0 -> 1 gradient, the minimal
    block-gradient of EPJ 2023 Eq. 9), block-gradient-block (BGBD,
    f_Gr = x_B) and diblock (BD, f_Gr = 0).
    Symmetric (x_B = 0.5): gradient, block-gradient-block (f_Gr = 0.5) and
    diblock. A block-gradient target does not exist for a symmetric
    composition: an A block followed by an A->B gradient cannot contain 50% B.
    """
    if symmetric:
        return STANDARD_FGR
    return (1.0, 2.0 * x_B, x_B, 0.0)


# ---------------------------------------------------------------------------
# Per-chain SD (Eqs. S11-S12) and normalisation (Eq. S14)
# ---------------------------------------------------------------------------

def _readings(S: np.ndarray, P: np.ndarray, symmetric: bool) -> np.ndarray:
    """SD* for each row of S (0 = A, 1 = B; equal length) against target P."""
    w = S.shape[1]
    T, Tr = np.cumsum(P), np.cumsum(P[::-1])
    S = S.astype(np.int32)

    def left(X):   # Eq. S11; |dS_A| = |dS_B| for binary chains, so one term suffices
        return np.abs(np.cumsum(X, axis=1) - T).sum(axis=1) / w ** 2

    def right(X):  # the same comparison with chain and target read from the end
        return np.abs(np.cumsum(X[:, ::-1], axis=1) - Tr).sum(axis=1) / w ** 2

    vals = [left(S), right(S)]
    if symmetric:
        vals += [left(1 - S), right(1 - S)]
    return np.min(vals, axis=0)                                    # Eq. S12


def homopolymer_reference(P: np.ndarray) -> float:
    """SD*_HP: the B-homopolymer read right->left against the target (Eq. S14)."""
    w = len(P)
    return float(np.abs(np.arange(1, w + 1) - np.cumsum(P[::-1])).sum() / w ** 2)


def chain_sd(sequence: np.ndarray, P: np.ndarray, symmetric: bool) -> float:
    """Normalised SD of one chain (0 = A, 1 = B) against target P of the same length."""
    s = np.asarray(sequence).reshape(1, -1)
    return float(_readings(s, P, symmetric)[0] / homopolymer_reference(P))


# ---------------------------------------------------------------------------
# Ideal reference sample and quality thresholds (EPJ 2023 Section 2.2, Eq. 21)
# ---------------------------------------------------------------------------

def ideal_sample(P: np.ndarray, n_chains: int = IDEAL_SAMPLE_SIZE,
                 seed: int = 0) -> np.ndarray:
    """
    Ideal sample of equal-length chains: at each position y exactly
    round(P_B(y) * n_chains) randomly chosen chains carry B (EPJ 2023, Fig. 3).
    """
    rng = np.random.default_rng(seed)
    w = len(P)
    ranks = np.argsort(rng.random((n_chains, w)), axis=0).argsort(axis=0)
    k = np.rint(P * n_chains).astype(int)
    return (ranks < k).astype(np.int8)


@functools.lru_cache(maxsize=256)
def ideal_thresholds(omega: int, x_B: float, f_Gr: float, symmetric: bool,
                     n_chains: int = IDEAL_SAMPLE_SIZE) -> tuple:
    """
    (Excellent/Good, Good/Poor) thresholds for one target, from its ideal
    sample scored with exactly the rules used for real chains.
    """
    def ideal_sd(f):
        P = target_probability(omega, x_B, f)
        return _readings(ideal_sample(P, n_chains), P, symmetric) / homopolymer_reference(P)

    sd = ideal_sd(f_Gr)
    exc_good = float(sd.mean())
    good_poor = float(np.quantile(ideal_sd(BLOCK_LIKE_FGR) if f_Gr < BLOCK_LIKE_FGR else sd, 0.98))
    return exc_good, good_poor


# ---------------------------------------------------------------------------
# Ensemble evaluation
# ---------------------------------------------------------------------------

@dataclass
class SDMetrics:
    """Multi-angle SD statistics of an ensemble against one target."""
    name: str
    f_Gr: float
    sd_values: np.ndarray        # normalised SD per scored chain
    sd_mean: float               # <SD>
    sd_std: float                # sigma_SD (population, RCE 2023 Eq. 2)
    sd_skewness: float           # mu~3,SD (RCE 2023 Eq. 4)
    sd_cv: float                 # CV_SD = sigma / <SD>
    sd_star_mean: float          # <SD*> (non-normalised)
    threshold_exc_good: float
    threshold_good_poor: float
    frac_excellent: float
    frac_good: float
    frac_poor: float


@dataclass
class SDEvaluation:
    """Everything the GUI or a script needs about one SD evaluation."""
    x_B: float                   # B-Func (minority fraction) used for the targets
    b_monomer: int               # which monomer id (1 or 2) is B
    symmetric: bool              # 4-reading rule used
    n_chains_scored: int
    n_chains_total: int
    mean_length: float
    reference_dp: int            # chain length of the ideal reference sample
    targets: dict = field(default_factory=dict)   # name -> SDMetrics
    scan: list = field(default_factory=list)      # [(f_Gr, <SD>)]
    note: str = ""                                # e.g. why a template is absent

    @property
    def best_f_Gr(self) -> Optional[float]:
        return min(self.scan, key=lambda t: t[1])[0] if self.scan else None


def _stats(name, f_Gr, sd, sd_star, thresholds) -> SDMetrics:
    m, s = float(sd.mean()), float(sd.std())
    skew = float(np.mean(((sd - m) / s) ** 3)) if s > 0 else 0.0
    exc, poor = thresholds
    return SDMetrics(
        name=name, f_Gr=f_Gr, sd_values=sd, sd_mean=m, sd_std=s, sd_skewness=skew,
        sd_cv=s / m if m > 0 else 0.0, sd_star_mean=float(sd_star.mean()),
        threshold_exc_good=exc, threshold_good_poor=poor,
        frac_excellent=float(np.mean(sd <= exc)),
        frac_good=float(np.mean((sd > exc) & (sd <= poor))),
        frac_poor=float(np.mean(sd > poor)),
    )


def prepare_chains(chains: Sequence[Sequence[int]], max_chains: Optional[int] = 10_000,
                   seed: int = 42):
    """
    Validates a two-monomer ensemble (monomer ids 1 and 2) and converts it to
    0/1 arrays with B = minority monomer.

    Returns (chains01, x_B, b_monomer, n_total). Raises ValueError otherwise.
    """
    chains = [np.asarray(c, dtype=np.int16) for c in chains if len(c) >= 2]
    if not chains:
        raise ValueError("No chains of length >= 2.")
    ids = set(np.unique(np.concatenate(chains)).tolist())
    if not ids <= {1, 2} or len(ids) < 2:
        raise ValueError(
            "SD metrics are defined for two-monomer copolymers only "
            f"(found monomer types {sorted(ids)}).")
    n_total = len(chains)
    if max_chains and n_total > max_chains:
        idx = np.random.default_rng(seed).choice(n_total, size=max_chains, replace=False)
        chains = [chains[i] for i in sorted(idx)]
    frac_2 = float(np.mean(np.concatenate(chains) == 2))
    b_monomer = 2 if frac_2 <= 0.5 else 1
    chains01 = [(c == b_monomer).astype(np.int8) for c in chains]
    x_B = frac_2 if b_monomer == 2 else 1.0 - frac_2
    return chains01, x_B, b_monomer, n_total


def evaluate(chains: Sequence[Sequence[int]], f_grs: Optional[Sequence[float]] = None,
             scan: bool = True, max_chains: Optional[int] = 10_000,
             ideal_size: int = IDEAL_SAMPLE_SIZE) -> SDEvaluation:
    """
    Scores every chain (each against the target at its own length) for each
    target in ``f_grs`` (default: :func:`standard_targets` for the polymer's
    composition), with ideal-sample thresholds at the number-average
    chain length. With ``scan``, also reports <SD> for f_Gr = 0, 0.1, ..., 1
    (RCE 2023 Fig. 9): its minimum indicates the best-matching structure.
    """
    chains01, x_B, b_monomer, n_total = prepare_chains(chains, max_chains)
    symmetric = abs(x_B - 0.5) <= SYMMETRIC_TOLERANCE
    x_B = min(x_B, 0.5)
    by_len = {}
    for c in chains01:
        by_len.setdefault(len(c), []).append(c)
    groups = [(w, np.array(cs)) for w, cs in sorted(by_len.items())]
    mean_len = float(np.mean([len(c) for c in chains01]))
    ref_dp = max(2, int(round(mean_len)))

    def score(f):
        sd, sd_star = [], []
        for w, S in groups:
            P = target_probability(w, x_B, f)
            raw = _readings(S, P, symmetric)
            sd_star.append(raw)
            sd.append(raw / homopolymer_reference(P))
        return np.concatenate(sd), np.concatenate(sd_star)

    result = SDEvaluation(x_B=x_B, b_monomer=b_monomer, symmetric=symmetric,
                          n_chains_scored=len(chains01), n_chains_total=n_total,
                          mean_length=mean_len, reference_dp=ref_dp)
    xb_key = round(x_B, 3)
    if f_grs is None:
        f_grs = standard_targets(x_B, symmetric)
        if symmetric:
            result.note = ("No block-gradient template: for a 50/50 composition an A block "
                           "followed by an A→B gradient cannot contain half B, so the papers "
                           "define it only for asymmetric compositions.")
    for f in f_grs:
        sd, sd_star = score(f)
        thr = ideal_thresholds(ref_dp, xb_key, float(f), symmetric, ideal_size)
        name = f"{structure_name(x_B, f)} (f_Gr={f:.2f})"
        result.targets[name] = _stats(name, float(f), sd, sd_star, thr)
    if scan:
        result.scan = [(float(f), float(score(f)[0].mean())) for f in SCAN_FGR]
    return result
