"""
Fit two independent ridge regressions on the teammate-gap tables from clean.py:

    signed_gap_millis ~ driver

Each row is one teammate pair in one race: driver_a's time minus driver_b's
time, in milliseconds (positive = driver_a lost time to driver_b). Because
the target is already differenced within a team, the design matrix encodes
each driver with +1 in driver_a's column and -1 in driver_b's column (all
other drivers 0) rather than the more familiar one-hot "driver + constructor"
setup — this is what actually estimates a per-driver skill coefficient from
pairwise differences (a standard trick for these "linked pairwise comparison"
models, and the same structural idea the chained-ELO phase will build on).

No intercept: with a fully +1/-1 coded design and no constructor dummy, the
coefficients are directly interpretable as each driver's deviation from the
field-average pace (in ms) for that metric, given the ridge penalty holding
sparse/rarely-paired drivers in check.
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_pairs(filename):
    with open(DATA / filename, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fit_pairwise_ridge(rows, alphas=(0.1, 1, 3, 10, 30, 100, 300, 1000)):
    drivers = sorted({r["driver_a"] for r in rows} | {r["driver_b"] for r in rows})
    idx = {d: i for i, d in enumerate(drivers)}

    X = np.zeros((len(rows), len(drivers)))
    y = np.zeros(len(rows))
    for i, r in enumerate(rows):
        X[i, idx[r["driver_a"]]] = 1.0
        X[i, idx[r["driver_b"]]] = -1.0
        y[i] = float(r["signed_gap_millis"])

    model = RidgeCV(alphas=alphas, fit_intercept=False)
    model.fit(X, y)

    n_pairings = {d: 0 for d in drivers}
    for r in rows:
        n_pairings[r["driver_a"]] += 1
        n_pairings[r["driver_b"]] += 1

    results = sorted(
        zip(drivers, model.coef_, [n_pairings[d] for d in drivers]),
        key=lambda t: t[1],
    )
    return results, model.alpha_


def print_ranking(title, results, chosen_alpha, unit_note):
    print(f"\n=== {title} (ridge alpha={chosen_alpha:g}) ===")
    print(f"{'driver':<20} {'coef (ms)':>10}  {'n races':>7}   {unit_note}")
    for driver, coef, n in results:
        sign = "faster" if coef < 0 else "slower"
        print(f"{driver:<20} {coef:>10.1f}  {n:>7}   {sign} than field avg")


if __name__ == "__main__":
    race_rows = load_pairs("race_gap_2023.csv")
    quali_rows = load_pairs("quali_gap_2023.csv")

    race_results, race_alpha = fit_pairwise_ridge(race_rows)
    quali_results, quali_alpha = fit_pairwise_ridge(quali_rows)

    print_ranking(
        "RACE PACE (gap-to-teammate at the flag)",
        race_results,
        race_alpha,
        "negative = faster than field average",
    )
    print_ranking(
        "QUALIFYING PACE (gap-to-teammate, best session reached)",
        quali_results,
        quali_alpha,
        "negative = faster than field average",
    )
