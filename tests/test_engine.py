import numpy as np
import math

import pytest

from compositional_suite import analysis, simulation_engine as se


def _engine(**kw):
    p = {"N_MONs": 2, "N_CHAINs": 300, "R_values": [[0.52], [0.46]], "Ctr": 10.0, "seed": 11}
    p.update(kw)
    return se.SimulationEngine(p)


def test_seeded_run_is_bit_reproducible():
    """Golden values: any change that alters the RNG stream or propagation shows up here.
    If the change is intended, bump ENGINE_VERSION and update these numbers."""
    e = _engine()
    e.initialize_chains([50, 50], 60, 80)
    e.add_new_block([20, 80], 40, 90, purge_monomers=True)
    chains = [p.asArray() for p in e.polymerArray]
    d = analysis.calculate_pdi(chains)
    assert d["Mn_DP"] == 84.2
    assert d["PDI"] == pytest.approx(1.076466506056725, abs=1e-15)
    assert sum(c.count(1) for c in chains) == 9690
    assert chains[0][:12] == [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1]


def test_block_history_records_what_ran():
    e = _engine()
    e.initialize_chains([50, 50], 60, 80)
    e.add_new_block([20, 80], 40, 90, purge_monomers=True)
    assert [b["dp"] for b in e.block_history] == [60, 40]
    assert e.block_history[1]["mrs"] == [20.0, 80.0]
    assert e.block_history[1]["purge"] is True
    assert all(b["status"] == "ok" for b in e.block_history)


def test_dpn_matches_seeded_offset():
    # Each chain is seeded with one monomer from the pool, so DPn after block 1
    # is 1 + X*(DP-1), not X*DP. (Manuscript §4.3 uses this exact prediction.)
    e = _engine(N_CHAINs=200, Ctr=se.IDEAL_EXCHANGE)
    e.initialize_chains([50, 50], 60, 60)
    mn = analysis.calculate_pdi([p.asArray() for p in e.polymerArray])["Mn_DP"]
    assert mn == pytest.approx(1 + 0.6 * 59, abs=0.05)


def test_feed_length_is_validated():
    e = _engine()
    with pytest.raises(ValueError):
        e.initialize_chains([50, 30, 20], 20, 50)


@pytest.mark.parametrize("ctr", [10.0, 100.0])  # closed form is precise here; drifts below ~10 (see manuscript §4.2)
def test_dispersity_follows_transfer_law(ctr):
    e = _engine(N_CHAINs=1500, Ctr=ctr, seed=3)
    e.initialize_chains([50, 50], 100, 50)
    d = analysis.calculate_pdi([p.asArray() for p in e.polymerArray])["PDI"]
    assert d == pytest.approx(se.dispersity_from_ctr(ctr, 100, 50), abs=0.04)


def test_ctr_inversion_round_trip():
    ctr = se.ctr_for_target_dispersity(1.2, 100, 70)
    assert se.dispersity_from_ctr(ctr, 100, 70) == pytest.approx(1.2)
    assert se.ctr_for_target_dispersity(1.0001, 100, 70) is None   # below the floor
    assert math.isinf(se.IDEAL_EXCHANGE)


def test_check_r_values_flags_zero_in_multimonomer_row():
    w = se.check_r_values([[0.72, 1.0, 0.8], [1.04, 0.9, 0.8], [0.4, 0.2, 0.3], [0.0, 0.9, 1.1]])
    assert len(w) == 1 and "r_41 = 0" in w[0] and "r_42, r_43" in w[0]
    assert se.check_r_values([[0.0], [0.5]]) == []        # binary r = 0 is legitimate
    assert any("negative" in m for m in se.check_r_values([[-0.1], [1.0]]))


def test_non_homopropagating_monomer_follows_feed():
    """A row of zeros (k_ii = 0, cross rates unknown) chooses by concentration (engine 2.1)."""
    from collections import Counter
    e = se.SimulationEngine({"N_MONs": 3, "N_CHAINs": 600, "R_values": [[1, 1], [1, 1], [0, 0]],
                             "Ctr": se.IDEAL_EXCHANGE, "seed": 1})
    e.initialize_chains([90, 5, 5], 40, 50)
    after = Counter(c[i + 1] for c in (p.asArray() for p in e.polymerArray)
                    for i in range(len(c) - 1) if c[i] == 3)
    assert after[1] / (after[1] + after[2]) > 0.85


def test_kappa_star_and_eta_reference_values():
    rng = np.random.default_rng(0)
    cycle = [[(i + k) % 3 + 1 for i in range(120)] for k in range(60)]
    k = analysis.calculate_kappa_metrics(cycle)
    assert k["kappa_star"] == pytest.approx(-1.0, abs=0.02) and k["kappa"] == pytest.approx(-0.5, abs=0.02)
    assert analysis.calculate_uncertainty_coefficient(cycle) == pytest.approx(1.0, abs=1e-6)
    rand = [list(rng.choice([1, 2], size=120, p=[0.7, 0.3])) for _ in range(300)]
    assert abs(analysis.calculate_kappa_metrics(rand)["kappa_star"]) < 0.03
    assert analysis.calculate_uncertainty_coefficient(rand) < 0.01
