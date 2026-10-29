"""
Structural deviation (SD) metrics against Conka et al.:
EPJ 2023 (+SI Eqs. S1-S14) and RCE 2023. Reference values are quoted in
compositional_suite/sd_metrics/core.py.
"""
import numpy as np
import pytest

from compositional_suite import simulation_engine as se
from compositional_suite import sd_metrics as sd


def _simulate(rA, rB, feed=(50, 50), n=2000, dp=100, seed=1):
    e = se.SimulationEngine({"N_MONs": 2, "N_CHAINs": n, "R_values": [[rA], [rB]],
                             "Ctr": se.IDEAL_EXCHANGE, "seed": seed})
    e.initialize_chains(list(feed), dp, 100.0)
    return [p.asArray() for p in e.polymerArray]


def _gradient(ev):
    return next(m for m in ev.targets.values() if m.f_Gr == 1.0)


def test_homopolymer_reference_matches_si_eq_s14():
    # EPJ 2023 Fig. 9b: <GD*_HP> = 0.167 (symmetric gradient), <BD*_HP> = 0.125 (diblock)
    assert sd.homopolymer_reference(sd.target_probability(100, 0.5, 1.0)) == pytest.approx(0.167, abs=0.003)
    assert sd.homopolymer_reference(sd.target_probability(100, 0.5, 0.0)) == pytest.approx(0.125, abs=0.005)


@pytest.mark.parametrize("x_B", [0.5, 0.3, 0.1])
@pytest.mark.parametrize("f_Gr", [1.0, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1, 0.0])
def test_targets_are_probabilities_with_the_polymer_composition(x_B, f_Gr):
    p = sd.target_probability(100, x_B, f_Gr)
    assert p.min() >= 0 and p.max() <= 1
    assert p.mean() == pytest.approx(x_B, abs=0.011)


def test_asymmetric_tapered_targets_follow_si_s7_s8():
    s7 = sd.target_probability(100, 0.3, 0.8)          # f_Gr >= 2 x_B: A block + gradient
    assert s7[:20].max() == 0 and s7[-1] == pytest.approx(0.75)     # no pure-B end block
    s8 = sd.target_probability(100, 0.3, 0.2)          # f_Gr < 2 x_B: A block, gradient, B block
    assert s8[59] == 0 and s8[60] == pytest.approx(0.05) and s8[79] == 1   # ramp starts at 0
    assert sd.structure_name(0.3, 0.8) == "block-gradient"
    assert sd.structure_name(0.3, 0.2) == "block-gradient-block"


def test_ideal_thresholds_symmetric_gradient_dp100():
    # Self-consistent values; the published 0.12 / 0.30 were derived from rounded inputs.
    exc, poor = sd.ideal_thresholds(100, 0.5, 1.0, True)
    assert exc == pytest.approx(0.10, abs=0.01) and poor == pytest.approx(0.26, abs=0.015)
    exc_b, poor_b = sd.ideal_thresholds(100, 0.5, 0.0, True)
    assert exc_b == 0.0 and poor_b > 0                   # block-like rule: value at f_Gr = 0.25


def test_kinetic_gradient_matches_rce_reference():
    # RCE 2023 / ACS Polym. Au 2026 P1 (rA = 3, rB = 0.1): <GD> 0.14, sigma 0.08, skew +0.80
    g = _gradient(sd.evaluate(_simulate(3, 0.1)))
    assert 0.10 <= g.sd_mean <= 0.17 and g.sd_skewness > 0


def test_kinetic_random_copolymer_is_recognised():
    # Reference (rA = rB = 1): <GD> 0.38, sigma 0.06, negative skewness
    g = _gradient(sd.evaluate(_simulate(1, 1)))
    assert 0.30 <= g.sd_mean <= 0.42 and g.sd_skewness < 0 and g.sd_cv < 0.3


def test_result_does_not_depend_on_monomer_labels():
    a = _gradient(sd.evaluate(_simulate(3, 0.1))).sd_mean
    b = _gradient(sd.evaluate(_simulate(0.1, 3))).sd_mean
    assert a == pytest.approx(b, abs=0.02)


def test_minority_monomer_is_b():
    ev = sd.evaluate(_simulate(1, 1, feed=(30, 70), n=500), scan=False)
    assert ev.b_monomer == 1 and ev.x_B == pytest.approx(0.30, abs=0.02) and not ev.symmetric


def test_every_chain_is_scored():
    chains = _simulate(1, 1, n=800)
    ev = sd.evaluate(chains, scan=False)
    assert ev.n_chains_scored == len([c for c in chains if len(c) >= 2])


@pytest.mark.parametrize("chains", [[[1, 2, 3, 1]] * 20, [[1, 1, 1]] * 20])
def test_only_two_monomer_copolymers(chains):
    with pytest.raises(ValueError, match="two-monomer"):
        sd.evaluate(chains)


def test_asymmetric_compositions_get_all_four_conka_targets():
    ev = sd.evaluate(_simulate(3, 0.1, feed=(70, 30), n=600), scan=False)
    names = [n.split(" (")[0] for n in ev.targets]
    assert names == ["gradient", "block-gradient", "block-gradient-block", "diblock"]
    sym = sd.evaluate(_simulate(3, 0.1, n=600), scan=False)
    assert [n.split(" (")[0] for n in sym.targets] == ["gradient", "block-gradient-block", "diblock"]
    assert "block-gradient" in sym.note
