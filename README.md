# F1 True Driver Ability Measure

A data analytics project estimating Formula 1 driver skill independent of
car performance, using two independent statistical models cross-checked
against each other, across twelve seasons (2014-present) of race data.

**Research question:** Who is actually the best driver on the grid, once
you control for who has the best car?

---

## Quick Start

- **Just want the results?** Open `F1_Driver_Rankings_2014_present.xlsx` —
  a Combined sheet ranking all 63 drivers across all four models, plus one
  full-detail sheet per model.
- **Want to run the pipeline?** See `/src` — run the scripts listed under
  Reproduce below, in order.
- **Want the full write-up?** Keep reading for methodology, findings, and
  limitations below.

---

## Scope

- **Drivers covered:** every F1 driver who raced 2014-present (63 total)
- **Metrics:** race pace and qualifying pace, modeled separately
- **Models:** chained ELO (a rolling 1500-centered rating) and pairwise
  ridge regression (a career-average pace coefficient in milliseconds)
- **Comparison method:** teammates only — the one matchup in F1 where car
  quality is held constant, since both drivers are in the same machinery
- **Focus:** relative driver skill, not car performance. This project
  deliberately does not attempt to measure absolute car pace, does not
  adjust for team orders, and does not include lap-level strategy data
  (see Limitations)
- **Time window:** 2014-present, the current turbo-hybrid regulation era —
  chosen for regulatory consistency, while still spanning enough seasons
  for driver movement between teams to statistically connect the whole
  grid (see Methodology)

A single-season (2023-only) version was attempted first and discarded —
see Methodology below for why.

---

## Methodology

### The core idea: compare teammates, not the field

Wins, poles, and championships are all contaminated by car quality — a
brilliant driver in a mediocre car will lose, every time, to a mediocre
driver in a brilliant car. The one comparison that isn't contaminated is
two teammates in the same race, in (as close as F1 allows) identical
machinery. Every model here is built on that single idea: score each
driver only against the teammates they actually raced.

### Why a single-season model doesn't work

The first version fit a ridge regression on just the 2023 season. It
ranked Fernando Alonso almost as high as Max Verstappen — which looked
like a bold finding until checking what produced it. Since almost nobody
switched teams mid-season, the ten team pairs were ten separate,
disconnected islands of data (Red Bull's pair had zero data connecting it
to Aston Martin's pair). Ridge regression's L2 penalty anchors any
isolated pair symmetrically around zero, regardless of which pair it is —
so Alonso's margin over Lance Stroll and Verstappen's margin over Sergio
Perez both landed near zero independently, making them look comparable
purely by coincidence, not because the data ever actually compared them.

**Fix:** pool multiple seasons. Drivers change teams over the years —
Daniel Ricciardo alone went Red Bull → Renault → McLaren → AlphaTauri —
and every move links two previously-disconnected team pairs together
through the driver who switched. Pool enough seasons and the isolated
islands become one connected graph where every driver is comparable to
every other. This is why the project spans 2014-present rather than one
season, and it's what "chained" refers to in "chained ELO."

### Model 1: Chained ELO — current form

Every driver starts at rating 1500. After each race, both teammates'
ratings update based on who beat whom:

```
expected_A = 1 / (1 + 10^((R_B − R_A) / 400))
new_R_A    = R_A + K × (actual_A − expected_A)
```

`K` (set to 24, the standard chess default) controls swing size. An
upset — a lower-rated driver beating a higher-rated one — moves ratings
more than an expected result, since it's more informative. Run in
chronological order across 260 races, this produces a rolling read on a
driver's current level, not their career average.

### Model 2: Ridge regression — career-average edge

Every row is one teammate pairing in one race. `X` has one column per
driver (`+1`/`-1` depending on which side of the pairing they were on,
`0` otherwise); `y` is the millisecond gap. The model solves for one
number per driver that best explains every row at once:

```
β = (XᵀX + αI)⁻¹Xᵀy
```

`αI` is a penalty that stops drivers with tiny samples (a one-race
substitute, say) from getting an overconfident number off a small,
noisy sample. `α` is chosen automatically by cross-validation.

Race and qualifying pace are modeled separately in both cases — four
models total — because one-lap qualifying execution and race-day
racecraft/tire management/overtaking are different skills.

### Data cleaning rules

- **Race pace:** a team pair only counts if both teammates finished on
  the lead lap with a valid time gap. Retirements and lapped cars are
  excluded rather than backfilled with a guessed penalty (no flat "DNF =
  lost by 10 seconds" rule — it can't distinguish a mechanical failure
  from a driving error, and the penalty size would be an arbitrary
  number quietly shaping the results).
- **Qualifying pace:** gap is computed from each driver's best Q1/Q2/Q3
  time, not F1DB's own precomputed gap column, which is only populated
  for drivers who reached Q3.
- **Source:** [F1DB](https://github.com/f1db/f1db), a free,
  community-maintained CSV database covering every F1 season since 1950
  (chosen over OpenF1, whose historical coverage only starts in 2023 —
  too short a window to chain across).

After cleaning: 1,071 usable race-pace comparisons and 2,569 usable
qualifying comparisons, across 260 race weekends.

---

## Key findings

Top of the combined rankings (average rank across all four models,
1 = best of 63):

| Driver | Race ELO rank | Quali ELO rank | Race ridge rank | Quali ridge rank | Avg rank |
|---|---|---|---|---|---|
| Max Verstappen | 1 | 1 | 1 | 1 | 1.0 |
| Alexander Albon | 5 | 17 | 8 | 2 | 8.0 |
| Lewis Hamilton | 7 | 13 | 17 | 43 | 20.0 |
| Daniel Ricciardo | 44 | 43 | 3 | 6 | 24.0 |

- **Verstappen is #1 in all four rankings independently** — the most
  robust result in the project, precisely because it holds regardless of
  which model or metric is used.
- **Ricciardo is the clearest case of the two models disagreeing, and
  why that's useful.** Ridge (his whole-career average) ranks him 3rd for
  race pace and 6th for qualifying. ELO (his current-form rating) ranks
  him 44th and 43rd. Ridge is answering "how good has he been on
  average"; ELO is answering "where does his rating sit right now" — a
  rough 2019-2022 stretch at Renault/McLaren left a dent his brief 2023
  return never had the chance to repay.
- **Albon outperforms his public reputation.** Ranked 5th (race ELO) and
  2nd of 63 (qualifying ridge) — the model only sees his gap to actual
  teammates (Verstappen at Red Bull, then a difficult Williams), not the
  "couldn't handle a Red Bull seat" narrative around him.
- **Vettel, Räikkönen, and Perez rank in the bottom half** (average rank
  ~32nd, ~42nd, and ~38th of 63). Vettel's and Räikkönen's most dominant
  seasons predate this project's 2014 start, so their numbers reflect
  only the back half of each career. Perez's sample is dominated by three
  seasons as Verstappen's teammate during Red Bull's most dominant
  stretch in the sport's history — a difficult comparison for nearly
  anyone. None of this means the model is wrong; it means these numbers
  measure a specific, dated window of a career, not a lifetime reputation.

---

## Known limitations (read before citing any single number)

- **ELO has no recency weighting.** A slump from years ago and a slump
  from last month count identically — nothing "expires." This is what
  drives the Ricciardo gap above, and it also pulls Hamilton's qualifying
  ELO rank down to 43rd — well below his 13th-place ridge rank — since a
  long uneven career average includes both his most dominant years and a
  rougher recent stretch, weighted the same. A recency-weighted ELO
  variant is the natural next step and hasn't been built yet.
- **Ridge and ELO measure genuinely different things** (career average
  vs. current form) and are presented side by side rather than collapsed
  into one "true" score — meaning there's no single definitive number.
- **Small samples produce noisy, overconfident-looking numbers.** A
  3-race substitute stint gets a real coefficient/rating in these
  rankings, built on far less evidence than a 200-race career. Check the
  `n` (sample size) column before trusting any single rank.
- **Teammate comparison has real limits.** It can't measure absolute car
  performance, doesn't account for team orders (a driver told to hold
  position isn't fairly "beaten" that race), and a couple of isolated
  early-era backmarker-team clusters (drivers who never moved to a team
  still on the grid) don't connect to the main graph at all.
- **No lap-level detail yet.** Everything here is finishing gaps and
  qualifying times — nothing about tire degradation, wet-weather skill,
  or in-race strategy execution. OpenF1's lap-by-lap telemetry (2023
  onward) is the natural next data layer.

---

## Repository structure

```
f1-driver-ability/
├── data/raw/              F1DB's downloaded CSVs (gitignored — re-download, see below)
├── data/processed/        every intermediate and final table the pipeline produces
├── src/
│   ├── clean.py                 single-season (2023) teammate-gap tables — original MVP
│   ├── clean_multiseason.py     same, pooled 2014-2023 (fair vs-ELO comparison)
│   ├── clean_full_era.py        same, pooled 2014-present (final scope)
│   ├── ridge_model.py           ridge regression, single season
│   ├── ridge_model_multiseason.py / ridge_model_full_era.py   same, pooled
│   ├── elo_model.py             chained ELO, race pace, 2014-present
│   ├── elo_model_quali.py       chained ELO, qualifying pace, 2014-present
│   ├── compare_quali_vs_race_elo.py   same model, two metrics — do they agree?
│   ├── compare_ridge_vs_elo.py        two models, same metric — do they agree?
│   └── master_rankings.py / build_rankings_workbook.py   combines everything into one spreadsheet
└── F1_Driver_Rankings_2014_present.xlsx   the results, ready to open
```

## How to reproduce this analysis

1. Download F1DB's CSV release into `data/raw/` (see the
   [F1DB repo](https://github.com/f1db/f1db) for the current release link).
2. Run, in order:
   ```
   python3 src/clean_full_era.py
   python3 src/ridge_model_full_era.py
   python3 src/elo_model.py
   python3 src/elo_model_quali.py
   python3 src/master_rankings.py
   python3 src/build_rankings_workbook.py
   ```
3. Open `F1_Driver_Rankings_2014_present.xlsx` for the results.

---

## Tech stack / skills demonstrated

- **Data engineering:** ingesting and cleaning multi-season CSV data,
  with explicit, documented exclusion rules (DNF/lapped-car handling,
  missing qualifying sessions) rather than silent defaults
- **Statistics:** L2-regularized (ridge) regression solved via matrix
  algebra, an ELO rating system adapted from chess to sequential race
  data, correlation analysis between independent models
- **Graph validation:** connected-components analysis to detect and fix
  a real bug (disconnected comparison groups producing misleading
  results) before trusting any output
- **Python:** pandas/numpy/csv, scikit-learn (`RidgeCV`), openpyxl for a
  formula-driven Excel deliverable
- **Methodology transparency:** every non-obvious decision (why 2014, why
  teammate comparison, why two models instead of one, why DNFs are
  excluded rather than penalized) is documented at the point it was made

If you spot something wrong or want to discuss the methodology,
[open an issue](../../issues).
