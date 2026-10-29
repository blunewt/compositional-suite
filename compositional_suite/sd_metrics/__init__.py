"""
sd_metrics: structural deviation (SD) metrics for binary (two-monomer)
copolymers, following Conka et al. (Eur. Polym. J. 2023 and its SI;
React. Chem. Eng. 2023; ACS Polym. Au 2026). See core.py for definitions.
"""

from .core import (
    BLOCK_LIKE_FGR,
    IDEAL_SAMPLE_SIZE,
    SCAN_FGR,
    STANDARD_FGR,
    SYMMETRIC_TOLERANCE,
    SDEvaluation,
    SDMetrics,
    chain_sd,
    evaluate,
    homopolymer_reference,
    ideal_sample,
    ideal_thresholds,
    prepare_chains,
    standard_targets,
    structure_name,
    target_probability,
)
