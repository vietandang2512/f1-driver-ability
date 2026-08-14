# F1 True Driver Ability Measure

Estimating driver skill separately from car performance, using two complementary approaches: a
ridge regression on teammate gaps, and a chained ELO rating built from teammate head-to-heads
across the 2014–present turbo-hybrid era.

## Status: phase 1 + 2 prototypes working

### Phase 1 — ridge regression (single season, `src/ridge_model.py`)
Fits `signed_gap ~ driver` separately for qualifying pace and race pace, using 2023 teammate
pairs (F1DB). **Finding:** a single season's teammate-only comparisons form a *disconnected*
graph — 2023 splits into 10 separate components (9 isolated pairs + one 4-driver AlphaTauri
group), so coefficients are only valid *within* a team pair, not as a grid-wide ranking. This
directly motivated phase 2.

### Phase 2 — chained ELO (2014–present, `src/elo_model.py`)
Processes all 260 races from 2014 onward in chronological order, updating driver ratings via a
standard ELO update (K=24) based on teammate head-to-head classification order. Because drivers
move between teams across 12+ seasons, the comparison graph "chains" together over time.

**Result:** the 60-driver graph collapses from 10 disconnected components (single-season) to
3 — one main component covering 55 drivers, plus two small residual clusters (Marussia and
Caterham, both teams that folded in 2014–2015 without any driver moving on to a team still on
the grid — real history, not a bug).

Top of the ranking (plausible: Verstappen, Alonso, Leclerc, Russell all near the top) and the
bottom (Latifi, Sargeant, Kubica's post-injury comeback, Stroll — all widely regarded as
below-average by F1 analysts) both pass a sanity check. One honest caveat: Hamilton ranks lower
(#7) than his reputation might suggest — plausibly because plain ELO has no recency weighting
or rating-drift correction, so a long career average can understate a driver whose sharpest
years (McLaren, pre-2014) fall outside this era window, or whose most recent seasons (Mercedes'
2022–2024 slump) pull the average down. Worth investigating further before treating rankings as
final — flagged as an open item, not smoothed over.

## Repo layout
```
data/
  raw/         F1DB CSV dump (race results, qualifying results, races calendar, drivers, constructors)
  processed/   cleaned teammate-gap tables + ELO comparison history
src/
  clean.py       builds race_gap_2023.csv / quali_gap_2023.csv (phase 1 inputs)
  ridge_model.py phase 1: ridge regression on teammate gaps (2023 only)
  elo_model.py   phase 2: chained ELO across 2014-present
```

## Running it
```bash
python3 src/clean.py        # builds the 2023 teammate-gap tables
python3 src/ridge_model.py  # phase 1 ridge regression + rankings
python3 src/elo_model.py    # phase 2 chained ELO + rankings + connectivity check
```

### Phase 2b — qualifying chained ELO (`src/elo_model_quali.py`) + comparison (`src/compare_quali_vs_race_elo.py`)
Same mechanism as the race chain, applied to qualifying classification (`positionNumber` from
`f1db-races-qualifying-results.csv`) instead of race results. Connectivity turned out even
better than race: **2 components instead of 3** (61 of 63 drivers in the main component;
only Bianchi/Chilton, from the defunct Marussia days, remain isolated).

Comparing the two chains' end-of-2023 ratings (snapshotted from each driver's last 2023
race/session, NOT their final 2026-inclusive rating — see the script for why that distinction
matters) for the 22 drivers active that season: **Pearson correlation 0.894** — strong overall
agreement, with Verstappen and Hamilton ranked identically #1/#2 in both. The disagreements are
the interesting part, and they line up with real driver reputations: Alonso ranks #4 in race
pace but only #8 in qualifying (matches his reputation as a racecraft/strategy specialist more
than a pure one-lap qualifier), while Russell ranks #3 in qualifying but #7 in race (matches his
reputation for strong one-lap pace with tougher races). One caveat: Liam Lawson shows the
largest swing (+7 race vs. quali rank) but that's built on only 3 race / 5 qualifying
comparisons in 2023 (his mid-season substitute stint), so it's a thin, noisy sample, not a
strong finding on its own.

### Phase 4 — ridge extended to 2014–2023 pooled (`src/clean_multiseason.py`, `src/ridge_model_multiseason.py`) + ridge-vs-ELO cross-validation (`src/compare_ridge_vs_elo.py`)
Extended ridge from single-season (2023) to pooled 2014–2023 data (not full 2014-present —
deliberately stops at 2023 so it's comparable to ELO's end-of-2023 snapshot, with no 2024-2026
data leaking in). **Confirms the predicted fix: connectivity went from 10 components (2023
alone) to 2 components** for both race and qualifying — pooling seasons fixes ridge's
disconnection problem the same way it fixed ELO's, via the same mechanism (driver transfers
linking teams across years).

Cross-validated against chained ELO (end-of-2023 snapshot, ranked within the same 22-driver
overlap for a fair comparison): **Pearson correlation 0.764 (race), 0.786 (qualifying)** —
meaningfully positive, with exact agreement at the top (Verstappen #1 in both, both metrics).
Weaker than the quali-vs-race ELO comparison's 0.894, which makes sense: that was the *same*
model on two related metrics, this is two *different* models on the same metric.

Standout divergence: **Daniel Ricciardo** — ridge ranks him 3rd (race) / 5th (quali) out of 22
using his whole 2014-2023 career; ELO ranks him 19th / 17th using his end-of-2023 rating. Not a
bug — the same phenomenon as the earlier Hamilton caveat, confirmed with a second, starker
example: ridge pools a driver's whole career as equally-weighted evidence (so Ricciardo's
strong 2014-2018 Red Bull years still count fully), while ELO has no recency weighting, so a
multi-year rough patch (Renault/McLaren, 2019-2022) never gets "paid back" by a brief, mixed
2023 return. Both numbers are legitimate — they answer different questions ("average career
skill" vs. "current rating given trajectory") — but it's a real limitation worth being explicit
about before treating either ranking as the final word.

## Open items (see project doc for full history of decisions)
- Investigate the recency-weighting gap properly (Hamilton + now Ricciardo both show it) —
  consider a recency-weighted or season-decayed ELO variant, or a rolling-window ridge, so
  "current form" and "career average" can both be reported deliberately instead of the choice
  being an accidental side effect of which model you picked.
- OpenF1 telemetry enrichment for 2023+ (phase 4 in the original roadmap, renumbered — see project doc).
- Automate + ship (GitHub Actions refresh, polished write-up).
- Consider extending both ridge and ELO to the FULL 2014-present (not stopping at 2023) now that
  the pipeline for pooling seasons is proven out.
