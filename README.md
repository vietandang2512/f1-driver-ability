# F1 True Driver Ability Measure

Who is the best driver on the grid once you control for who has the best
car? This project estimates driver skill independent of machinery across
twelve seasons (2014-present), using two independent statistical models
checked against each other.

---

## Quick start

- Results: open `F1_Driver_Rankings_2014_present.xlsx`. The Combined
  sheet ranks all 63 drivers across all four models, with a full-detail
  sheet per model.
- Pipeline: see `/src` and run the scripts under Reproduce below, in
  order.
- Full write-up: methodology, findings, and limitations are below.

---

## Scope

- Drivers: every F1 driver who raced 2014-present, 63 in total
- Metrics: race pace and qualifying pace, modeled separately
- Models: chained ELO (a rolling rating centered on 1500) and pairwise
  ridge regression (a career-average pace coefficient in milliseconds)
- Comparison method: teammates only, the one matchup in F1 where car
  quality is held constant
- Relative driver skill only. No attempt to measure absolute car pace, no
  adjustment for team orders, no lap-level strategy data (see
  Limitations)
- Time window: 2014-present, the turbo-hybrid regulation era. Chosen for
  regulatory consistency while still spanning enough seasons for driver
  transfers to statistically connect the whole grid (see Methodology)

A single-season (2023-only) version was built first and discarded. The
reason is in Methodology below.

---

## Methodology

### Compare teammates, not the field

Wins, poles, and championships are all contaminated by car quality. A
brilliant driver in a mediocre car loses every time to a mediocre driver
in a brilliant car. The one clean comparison is two teammates in the same
race, in as close to identical machinery as F1 allows. Every model here
scores each driver only against the teammates they actually raced.

### Why a single-season model fails

The first version fit a ridge regression on the 2023 season alone. It
ranked Fernando Alonso almost as high as Max Verstappen, which looked
like a bold finding until I checked what produced it. Almost nobody
switched teams mid-season, so the ten team pairs were ten disconnected
islands of data: nothing linked Red Bull's pair to Aston Martin's. Ridge
regression's L2 penalty anchors any isolated pair symmetrically around
zero regardless of which pair it is, so Alonso's margin over Lance Stroll
and Verstappen's margin over Sergio Perez both landed near zero
independently. They looked comparable by coincidence, not because the
data ever compared them.

The fix is pooling multiple seasons. Drivers change teams over the years
(Daniel Ricciardo alone went Red Bull → Renault → McLaren → AlphaTauri),
and every move links two previously disconnected team pairs through the
driver who switched. Pool enough seasons and the islands become one
connected graph where every driver is comparable to every other. That is
why the project spans 2014-present, and it is what "chained" means in
"chained ELO."

### Model 1: chained ELO, current form

Every driver starts at 1500. After each race, both teammates' ratings
update based on who beat whom:

```
expected_A = 1 / (1 + 10^((R_B − R_A) / 400))
new_R_A    = R_A + K × (actual_A − expected_A)
```

K is set to 24, the standard chess default, and controls swing size. An
upset moves ratings more than an expected result, because it carries more
information. Run chronologically across 260 races, this gives a rolling
read on a driver's current level rather than a career average.

### Model 2: ridge regression, career-average edge

Every row is one teammate pairing in one race. X has one column per
driver (+1 or -1 depending on which side of the pairing they were on, 0
otherwise) and y is the millisecond gap. The model solves for one number
per driver that best explains every row at once:

```
β = (XᵀX + αI)⁻¹Xᵀy
```

The αI term penalizes drivers with tiny samples, so a one-race substitute
can't earn an overconfident number off a handful of noisy rows. α is
chosen by cross-validation.

Race and qualifying pace are modeled separately in both cases, giving
four models, because one-lap qualifying execution and race-day racecraft,
tire management, and overtaking are different skills.

### Data cleaning

Race pace counts a team pair only when both teammates finished on the
lead lap with a valid time gap. Retirements and lapped cars are excluded
rather than backfilled with a guessed penalty. A flat "DNF = lost by 10
seconds" rule can't tell a mechanical failure from a driving error, and
the size of the penalty would be an arbitrary number quietly shaping the
results.

Qualifying pace uses each driver's best Q1/Q2/Q3 time to compute the gap,
not F1DB's precomputed gap column, which is only populated for drivers
who reached Q3.

Data comes from F1DB, a free community-maintained CSV database covering
every F1 season since 1950. OpenF1 was the alternative, but its
historical coverage starts in 2023, too short a window to chain across.

After cleaning: 1,071 usable race-pace comparisons and 2,569 usable
qualifying comparisons, across 260 race weekends.

---

## Findings

Top of the combined rankings, by average rank across all four models
(1 = best of 63):

| Driver | Race ELO | Quali ELO | Race ridge | Quali ridge | Avg rank |
|---|---|---|---|---|---|
| Max Verstappen | 1 | 1 | 1 | 1 | 1.0 |
| Alexander Albon | 5 | 17 | 8 | 2 | 8.0 |
| Lewis Hamilton | 7 | 13 | 17 | 43 | 20.0 |
| Daniel Ricciardo | 44 | 43 | 3 | 6 | 24.0 |

Verstappen ranks first in all four models independently. That makes it
the most robust result here, since it holds whichever model or metric you
pick.

Ricciardo is the sharpest disagreement between the two models, and the
disagreement is the point. Ridge, his whole-career average, puts him 3rd
for race pace and 6th for qualifying. ELO, his current-form rating, puts
him 44th and 43rd. Ridge answers how good he has been on average; ELO
answers where his rating sits now. A rough 2019-2022 stretch at Renault
and McLaren left a dent his brief 2023 return never had time to repay.

Albon outperforms his public reputation, ranking 5th on race ELO and 2nd
of 63 on qualifying ridge. The model sees only his gap to the teammates
he actually had (Verstappen at Red Bull, then a difficult Williams), not
the "couldn't handle a Red Bull seat" story around him.

Vettel, Räikkönen, and Perez land in the bottom half, averaging roughly
32nd, 42nd, and 38th of 63. Vettel's and Räikkönen's most dominant
seasons predate the 2014 start, so their numbers cover only the back half
of each career. Perez's sample is dominated by three seasons as
Verstappen's teammate during Red Bull's most dominant stretch in the
sport's history, a brutal comparison for nearly anyone. That doesn't make
the model wrong. It means these numbers measure a specific, dated window
of a career rather than a lifetime reputation.

---

## Limitations

Read these before citing any single number.

**ELO has no recency weighting.** A slump from years ago and a slump from
last month count the same, and nothing expires. This drives the Ricciardo
gap above, and it also pulls Hamilton's qualifying ELO to 43rd against a
13th-place ridge rank, since a long uneven career average weights his
most dominant years and his rougher recent stretch equally. A
recency-weighted ELO variant is the obvious next step and isn't built
yet.

**The two models measure different things**, career average versus
current form, and are presented side by side rather than collapsed into
one score. There is no single definitive number here.

**Small samples produce noisy, overconfident-looking results.** A
three-race substitute stint gets a real coefficient and rating built on
far less evidence than a 200-race career. Check the n column before
trusting any single rank.

**Teammate comparison has real limits.** It can't measure absolute car
performance, it ignores team orders (a driver told to hold position
wasn't fairly beaten that race), and a few isolated early-era backmarker
clusters, drivers who never moved to a team still on the grid, never
connect to the main graph.

**No lap-level detail.** Everything here is finishing gaps and qualifying
times, with nothing on tire degradation, wet-weather skill, or in-race
strategy. OpenF1's lap-by-lap telemetry from 2023 onward is the next data
layer.

---

## Repository structure

```
f1-driver-ability/
├── data/raw/              F1DB CSVs (gitignored, re-download - see below)
├── data/processed/        every intermediate and final table
├── src/
│   ├── clean.py                 single-season (2023) teammate-gap tables, original MVP
│   ├── clean_multiseason.py     same, pooled 2014-2023 (fair vs-ELO comparison)
│   ├── clean_full_era.py        same, pooled 2014-present (final scope)
│   ├── ridge_model.py           ridge regression, single season
│   ├── ridge_model_multiseason.py / ridge_model_full_era.py   same, pooled
│   ├── elo_model.py             chained ELO, race pace, 2014-present
│   ├── elo_model_quali.py       chained ELO, qualifying pace, 2014-present
│   ├── compare_quali_vs_race_elo.py   same model, two metrics: do they agree?
│   ├── compare_ridge_vs_elo.py        two models, same metric: do they agree?
│   └── master_rankings.py / build_rankings_workbook.py   combine into one spreadsheet
└── F1_Driver_Rankings_2014_present.xlsx   the results
```

## Reproducing the analysis

1. Download the F1DB CSV release into `data/raw/` (see the F1DB repo for
   the current release link).
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
