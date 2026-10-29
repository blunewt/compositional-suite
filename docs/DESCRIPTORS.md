# Sequence descriptors

Compositional Suite summarises the monomer sequences of a simulated (or
loaded) chain ensemble with three numbers. All three are computed over every
chain in the ensemble and are defined for any number of monomer types, so
copolymers, terpolymers and higher systems are placed on one scale. They are
adapted from the sequence descriptors of Guzman et al. [1].

## κ\* — neighbour correlation

Let *P*<sub>same</sub> be the fraction of neighbouring pairs whose two units
are the same monomer, and *C* = Σ *f*<sub>i</sub>² the fraction expected by
chance for a random sequence with the same composition *f*. Cohen's kappa [2]
applied to neighbouring units is

    κ = (P_same − C) / (1 − C)

κ is 0 for a random sequence and +1 when all neighbours are identical, but its
negative limit depends on composition. κ\* rescales the negative side to the
most alternating sequence the composition allows:

    κ* = κ                                  if P_same ≥ C
    κ* = (P_same − C) / (C − P_min)          otherwise

with *P*<sub>min</sub> = 0 if the largest monomer fraction is at most 0.5,
otherwise 2 *f*<sub>max</sub> − 1. Hence **−1 = as alternating as the
composition allows, 0 = random, +1 = blocky**, for any composition and number of
monomers. For two monomers at equal composition, κ\* equals the non-unit
eigenvalue of the terminal-model transition matrix.

## U — order

U is Theil's uncertainty coefficient [3] of the next unit given the current one,
over all neighbouring pairs:

    U = 1 − H(next | current) / H(next)

with Shannon entropies H. U is the fraction of the uncertainty about the next
unit that knowing the current unit removes: **0 for a random sequence, 1 when
every unit determines the next** (for example a strict cycle ABCABC…). U
separates ordered sequences from merely non-repeating ones that κ\* ranks
alike.

## ⟨L⟩ — mean run length

The average length of uninterrupted runs of one monomer, over all chains: an
absolute measure of blockiness (about 2 for a random 50:50 copolymer, 1 for a
strictly alternating one).

## Limits

All three are neighbour and run statistics. They do not record where along the
chain runs occur or how regularly they are spaced: a regular multiblock and a
statistical copolymer with the same mean run length have nearly identical
values. Positional composition profiles and structural deviation (two-monomer
systems) capture that information.

## References

1. G. Guzman, J. Ting, S. Murthy, T. Neenan, S. Kevlahan. *ChemRxiv* 2025,
   doi:10.26434/chemrxiv-2025-cs3gj.
2. J. Cohen. A coefficient of agreement for nominal scales. *Educ. Psychol.
   Meas.* **1960**, 20, 37–46.
3. H. Theil. On the estimation of relationships involving qualitative
   variables. *Am. J. Sociol.* **1970**, 76, 103–154.
