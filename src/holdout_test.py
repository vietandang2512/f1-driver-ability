"""
Accuracy test #2: chronological holdout.

Unlike ELO (which makes a fresh, leak-free prediction before every race by
design -- see elo_calibration.py), ridge regression solves for every
driver's coefficient in one joint calculation using ALL the data it's
given. That means ridge needs an explicit train/test split to check whether
it generalizes, the same way any regression model would: fit it on data up
through a cutoff, then see how well it predicts data it never saw.

The split used here: train on 2014-2022, test on 2023 -- a full season held
out entirely. Two things get checked on the 2023 holdout:

1. Sign accuracy: for every 2023 teammate pairing, did the model (fit only
   on 2014-2022) correctly predict WHICH of the two drivers would be
   faster? Baseline is 50%, since which driver is labeled "a" vs "b" in
   the data is arbitrary (alphabetical by driver ID), not tied to skill.
2. Mean absolute error (MAE): how far off, in milliseconds, was the
   predicted gap from the actual gap? Baseline is "always predict a 0ms
   gap" (i.e. no model at all) -- beating that baseline is the minimum bar
   for the model to be adding any value.

For comparison, chained ELO gets the same train/test treatment: ratings are
built from 2014-2022 only and then FROZEN (no further updates), and those
frozen ratings are used to predict every 2023 comparison. This is a
stricter test than elo_calibration.py's rolling walk-forward check, since
here the model isn't allowed to keep learning during 2023 at all -- it's
judged purely on whether last year's rating still predicts this year's
results.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA = Path(__file__).resolve().parent.parent / "data" / "processed"

TRAIN_END_YEAR = 2022  # inclusive
TEST_YEAR = 2023
ERA_START_YEAR = 2014
K_FACTOR = 24
INITIAL_RATING = 1500.0


# ---------- Ridge holdout ----------

def load_gap_rows(filename):
    with open(DATA / filename, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fit_pairwise_ridge(rows, alphas=(0.1, 1, 3, 10, 30, 100, 300, 1000, 3000)):
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
    return {d: c for d, c in zip(drivers, model.coef_)}


def ridge_holdout(label, filename):
    rows = load_gap_rows(filename)
    train_rows = [r for r in rows if int(r["year"]) <= TRAIN_END_YEAR]
    test_rows = [r for r in rows if int(r["year"]) == TEST_YEAR]

    coefs = fit_pairwise_ridge(train_rows)

    evaluable, skipped_unseen = [], 0
    for r in test_rows:
        a, b = r["driver_a"], r["driver_b"]
        if a not in coefs or b not in coefs:
            skipped_unseen += 1
            continue
        predicted_gap = coefs[a] - coefs[b]
        actual_gap = float(r["signed_gap_millis"])
        evaluable.append((predicted_gap, actual_gap))

    correct_sign = sum(1 for p, a in evaluable if (p > 0) == (a > 0))
    n = len(evaluable)
    sign_accuracy = correct_sign / n if n else float("nan")

    mae_model = sum(abs(p - a) for p, a in evaluable) / n if n else float("nan")
    mae_baseline = sum(abs(a) for _, a in evaluable) / n if n else float("nan")

    print(f"\n=== {label}: ridge holdout (train {ERA_START_YEAR}-{TRAIN_END_YEAR}, test {TEST_YEAR}) ===")
    print(f"Test rows: {len(test_rows)}  (evaluable: {n}, skipped -- driver debuted in {TEST_YEAR}: {skipped_unseen})")
    print(f"Sign accuracy (picked the actually-faster driver): {sign_accuracy:.1%}  (baseline: 50.0%)")
    print(f"MAE: {mae_model:,.0f} ms  (baseline -- always predict 0ms gap: {mae_baseline:,.0f} ms)")
    return sign_accuracy, mae_model, mae_baseline


# ---------- ELO holdout ----------

def load_raw_rows(filename):
    with open(RAW / filename, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def elo_holdout(label, filename):
    results = [
        r for r in load_raw_rows(filename) if r["year"] and int(r["year"]) >= ERA_START_YEAR
    ]
    by_race = defaultdict(list)
    for r in results:
        by_race[(int(r["year"]), int(r["round"]), r["raceId"])].append(r)
    race_keys = sorted(by_race.keys())

    ratings = defaultdict(lambda: INITIAL_RATING)

    # Phase 1: train through TRAIN_END_YEAR, updating ratings normally.
    for year, round_, race_id in race_keys:
        if year > TRAIN_END_YEAR:
            continue
        rows = by_race[(year, round_, race_id)]
        by_constructor = defaultdict(list)
        for r in rows:
            by_constructor[r["constructorId"]].append(r)
        for constructor_id, pair in by_constructor.items():
            classified = [r for r in pair if r["positionNumber"]]
            if len(classified) != 2:
                continue
            a, b = sorted(classified, key=lambda r: r["driverId"])
            pos_a, pos_b = int(a["positionNumber"]), int(b["positionNumber"])
            ra, rb = ratings[a["driverId"]], ratings[b["driverId"]]
            exp_a = expected_score(ra, rb)
            actual_a = 1.0 if pos_a < pos_b else 0.0
            ratings[a["driverId"]] = ra + K_FACTOR * (actual_a - exp_a)
            ratings[b["driverId"]] = rb + K_FACTOR * ((1 - actual_a) - (1 - exp_a))

    frozen_ratings = dict(ratings)  # snapshot -- NOT updated further below
    known_drivers = set(frozen_ratings)

    # Phase 2: predict TEST_YEAR using frozen ratings only, no updates.
    correct, total, skipped_unseen = 0, 0, 0
    for year, round_, race_id in race_keys:
        if year != TEST_YEAR:
            continue
        rows = by_race[(year, round_, race_id)]
        by_constructor = defaultdict(list)
        for r in rows:
            by_constructor[r["constructorId"]].append(r)
        for constructor_id, pair in by_constructor.items():
            classified = [r for r in pair if r["positionNumber"]]
            if len(classified) != 2:
                continue
            a, b = sorted(classified, key=lambda r: r["driverId"])
            if a["driverId"] not in known_drivers or b["driverId"] not in known_drivers:
                skipped_unseen += 1
                continue
            pos_a, pos_b = int(a["positionNumber"]), int(b["positionNumber"])
            ra, rb = frozen_ratings[a["driverId"]], frozen_ratings[b["driverId"]]
            exp_a = expected_score(ra, rb)
            actual_a = 1.0 if pos_a < pos_b else 0.0
            if exp_a == 0.5:
                continue  # no favorite to score
            total += 1
            if (exp_a > 0.5) == (actual_a == 1.0):
                correct += 1

    accuracy = correct / total if total else float("nan")
    print(f"\n=== {label}: ELO holdout (ratings frozen at end of {TRAIN_END_YEAR}, tested on {TEST_YEAR}) ===")
    print(f"Test comparisons: {total} (skipped -- driver debuted in {TEST_YEAR}: {skipped_unseen})")
    print(f"Favorite-picked-correctly: {accuracy:.1%}  (baseline: 50.0%)")
    return accuracy


if __name__ == "__main__":
    ridge_holdout("RACE PACE", "race_gap_2014_present.csv")
    ridge_holdout("QUALIFYING PACE", "quali_gap_2014_present.csv")
    elo_holdout("RACE PACE", "f1db-races-race-results.csv")
    elo_holdout("QUALIFYING PACE", "f1db-races-qualifying-results.csv")
