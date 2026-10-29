# Reactivity-ratio library

The program ships a curated library of terminal-model reactivity ratios,
compiled by the authors from the primary literature: 128 determinations for
47 monomers (73 monomer pairs), each with its reference (DOI where available).
It includes monomers used in our own work, such as acrylic and methacrylic
acid, maleic anhydride, saccharin methacrylamide, AMPS, NIPAM and charged
(meth)acrylates and methacrylamides.

| File | Contents |
|---|---|
| `compositional_suite/data/reactivity_ratios.csv` | curated library, 128 entries (127 usable) |

Browse it under **Help → Reactivity Ratio Library** and fill r_ij with
**From library…**. Users can add their own tables in the same CSV format
(**Help → Reactivity Ratio Library → Add library file…**); these are loaded
alongside the shipped library on that computer only.

Reactivity ratios depend on temperature, solvent, pH and fitting method: pick the
determination that matches your conditions and cite its primary reference.

## Columns

Required: `id`, `monomer_1`, `monomer_2`, `r1`, `r2`, `usable`, `dataset`.
Optional columns are shown in the library browser when present.

| Column | Meaning |
|---|---|
| `id` | unique entry id (`OWN-0001`… for the shipped library) |
| `monomer_1`, `abbr_1`, `monomer_2`, `abbr_2` | monomer names and abbreviations |
| `r1`, `r2` | reactivity ratios, r1 for monomer 1 |
| `r1_ci95`, `r2_ci95` | ±95 % confidence intervals, where reported |
| `temperature_C`, `solvent`, `conditions`, `method` | experimental conditions as reported |
| `model` | `terminal` (only terminal-model entries are used in simulations) |
| `usable` | `yes` if both ratios are present, non-negative and terminal-model |
| `flags` | semicolon-separated notes, e.g. `penultimate_model_fit` |
| `reference`, `doi` | primary reference and its DOI |
| `dataset` | source label (`own_curated` for the shipped library) |

## Default entry for a pair

Lookups are orientation-aware. Usable entries are ranked: unflagged first, then
the narrowest reported 95 % intervals relative to r, then the temperature closest
to 60 °C, then entries with a DOI. The first is proposed; every determination is
listed in the dialog. For three or more monomers the matrix is assembled pairwise
from binary entries.
