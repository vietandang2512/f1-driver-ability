"""
Accuracy test #1: ELO calibration.

Chained ELO already makes a concrete, checkable prediction before every
single race: "driver A has an exp_a probability of beating driver B this
race." That prediction is made using only ratings built from races BEFORE
this one -- so checking it against what actually happened is a legitimate,
leak-free accuracy test, with no need for a separate train/test split. This
script re-runs the exact same chained ELO loop as elo_model.py /
elo_model_quali.py, but additionally records the predicted probability for
every comparison, then checks two things:

1. Brier score -- the standard way to score a probability forecast. It's
   the mean squared error between the predicted probability and the actual
   outcome (1 or 0). Lower is better; 0.25 is what you'd get from a model
   that always guesses 50/50 (a "no information" baseline), and 0.0 would
   be a perfect, always-certain-and-always-right model -- which is not a
   realistic target here since race outcomes are genuinely uncertain.

2. Calibration table -- of every race where the model predicted, say,
   "60-70% chance A wins," did A actually win about 60-70% of the time?
   Good calibration means yes. If the model is systematically overconfident
   (predicts 90% but the favorite only wins 70% of the time) or
   underconfident (predicts 60% but the favorite wins 85% of the time),
   this table will show it directly.
"""

import csv
from collections import defaultdict
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

ERA_START_YEAR = 2014
K_FACTOR = 24
INITIAL_RATING = 1500.0


def load_rows(filename):
    with open(RAW / filename, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def run_chained_elo_with_predictions(filename):
    """Same loop as elo_model.py, but records the pre-race prediction
    (exp_a) and the actual outcome (actual_a) for every comparison."""
    results = [
        r for r in load_rows(filename) if r["year"] and int(r["year"]) >= ERA_START_YEAR
    ]

    by_race = defaultdict(list)
    for r in results:
        by_race[(int(r["year"]), int(r["round"]), r["raceId"])].append(r)

    race_keys = sorted(by_race.keys())
    ratings = defaultdict(lambda: INITIAL_RATING)
    predictions = []  # one row per comparison: year, round, exp_a, actual_a

    for year, round_, race_id in race_keys:
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

            predictions.append(
                {
                    "year": year,
                    "round": round_,
                    "driver_a": a["driverId"],
                    "driver_b": b["driverId"],
                    "predicted_prob_a": exp_a,
                    "actual_a_won": actual_a,
                }
            )

            new_ra = ra + K_FACTOR * (actual_a - exp_a)
            new_rb = rb + K_FACTOR * ((1 - actual_a) - (1 - exp_a))
            ratings[a["driverId"]] = new_ra
            ratings[b["driverId"]] = new_rb

    return predictions


def brier_score(predictions):
    n = len(predictions)
    return sum((p["predicted_prob_a"] - p["actual_a_won"]) ** 2 for p in predictions) / n


def pick_favorite_accuracy(predictions):
    """Of all comparisons where the model had a clear favorite (prob != 0.5),
    how often did that favorite actually win?"""
    decisive = [p for p in predictions if p["predicted_prob_a"] != 0.5]
    correct = sum(
        1
        for p in decisive
        if (p["predicted_prob_a"] > 0.5) == (p["actual_a_won"] == 1.0)
    )
    return correct / len(decisive), len(decisive)


def calibration_table(predictions, n_bins=10):
    """Bucket predictions by predicted probability (mapped to 'favorite
    confidence', i.e. always >=0.5) and compare to actual favorite win rate
    in each bucket."""
    buckets = defaultdict(lambda: {"n": 0, "wins": 0, "prob_sum": 0.0})
    for p in predictions:
        # fold to the favored side so buckets read as "confidence level"
        prob = p["predicted_prob_a"]
        favored_won = p["actual_a_won"] == 1.0
        if prob < 0.5:
            prob = 1 - prob
            favored_won = not favored_won
        bin_idx = min(int(prob * n_bins), n_bins - 1)
        b = buckets[bin_idx]
        b["n"] += 1
        b["wins"] += 1 if favored_won else 0
        b["prob_sum"] += prob
    return buckets


def print_report(label, predictions):
    n = len(predictions)
    bs = brier_score(predictions)
    acc, n_decisive = pick_favorite_accuracy(predictions)
    baseline_bs = 0.25  # always-guess-50/50 baseline

    print(f"\n=== {label}: ELO calibration report ({n} comparisons, {ERA_START_YEAR}-present) ===")
    print(f"Brier score: {bs:.4f}  (lower is better; 0.25 = always-guess-50/50 baseline)")
    print(f"Favorite-picked-correctly: {acc:.1%} of {n_decisive} decisive comparisons "
          f"(baseline for a coin flip: 50.0%)")

    print(f"\n{'confidence bucket':<20}{'n':>6}{'predicted avg':>16}{'actual win rate':>18}")
    buckets = calibration_table(predictions)
    for bin_idx in sorted(buckets):
        b = buckets[bin_idx]
        lo, hi = bin_idx * 10, bin_idx * 10 + 10
        predicted_avg = b["prob_sum"] / b["n"]
        actual_rate = b["wins"] / b["n"]
        print(f"{lo:>3}-{hi:<3}%{'':<12}{b['n']:>6}{predicted_avg:>15.1%}{actual_rate:>18.1%}")


def write_predictions(predictions, filename):
    path = OUT / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)
    print(f"wrote {len(predictions)} rows -> {path}")


if __name__ == "__main__":
    race_predictions = run_chained_elo_with_predictions("f1db-races-race-results.csv")
    print_report("RACE PACE", race_predictions)
    write_predictions(race_predictions, "elo_calibration_race.csv")

    quali_predictions = run_chained_elo_with_predictions("f1db-races-qualifying-results.csv")
    print_report("QUALIFYING PACE", quali_predictions)
    write_predictions(quali_predictions, "elo_calibration_quali.csv")
