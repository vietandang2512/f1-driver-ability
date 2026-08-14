"""
Combines all four full-era (2014-present) rankings -- race ELO, qualifying
ELO, race ridge, qualifying ridge -- into one master CSV, one row per driver,
with each model's raw score and rank. Drivers not covered by a given model
(e.g. too few classified races for ridge's stricter data needs) get blanks
for that model rather than being dropped from the table entirely.
"""

import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_elo_final_ratings(history_filename):
    """Each driver's LAST rating in the full history (i.e. their final 2014-present rating)."""
    latest = {}  # driver -> (year, round, rating)
    with open(DATA / history_filename, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year, round_ = int(row["year"]), int(row["round"])
            for side in ("a", "b"):
                driver = row[f"driver_{side}"]
                rating = float(row[f"rating_{side}_after"])
                key = (year, round_)
                if driver not in latest or key > latest[driver][0]:
                    latest[driver] = (key, rating)
    races_seen = {}
    with open(DATA / history_filename, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for side in ("a", "b"):
                d = row[f"driver_{side}"]
                races_seen[d] = races_seen.get(d, 0) + 1
    return {d: r for d, (_, r) in latest.items()}, races_seen


def load_ridge_coefs(filename):
    coefs, n = {}, {}
    with open(DATA / filename, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            coefs[row["driver"]] = float(row["coef_ms"])
            n[row["driver"]] = int(row["n_pairings"])
    return coefs, n


def rank_map(values, ascending):
    ranked = sorted(values.items(), key=lambda t: t[1], reverse=not ascending)
    return {d: i + 1 for i, (d, _) in enumerate(ranked)}


if __name__ == "__main__":
    race_elo, race_elo_n = load_elo_final_ratings("chained_elo_history_2014_present.csv")
    quali_elo, quali_elo_n = load_elo_final_ratings("chained_elo_history_quali_2014_present.csv")
    race_ridge, race_ridge_n = load_ridge_coefs("ridge_race_coefs_2014_present.csv")
    quali_ridge, quali_ridge_n = load_ridge_coefs("ridge_quali_coefs_2014_present.csv")

    race_elo_rank = rank_map(race_elo, ascending=False)
    quali_elo_rank = rank_map(quali_elo, ascending=False)
    race_ridge_rank = rank_map(race_ridge, ascending=True)   # lower ms = faster
    quali_ridge_rank = rank_map(quali_ridge, ascending=True)

    all_drivers = sorted(set(race_elo) | set(quali_elo) | set(race_ridge) | set(quali_ridge))
    print(f"Total distinct drivers across all four models: {len(all_drivers)}")

    out_path = DATA / "master_rankings_2014_present.csv"
    fieldnames = [
        "driver",
        "race_elo_rating", "race_elo_rank", "race_elo_n",
        "quali_elo_rating", "quali_elo_rank", "quali_elo_n",
        "race_ridge_ms", "race_ridge_rank", "race_ridge_n",
        "quali_ridge_ms", "quali_ridge_rank", "quali_ridge_n",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in all_drivers:
            writer.writerow({
                "driver": d,
                "race_elo_rating": round(race_elo[d], 1) if d in race_elo else "",
                "race_elo_rank": race_elo_rank.get(d, ""),
                "race_elo_n": race_elo_n.get(d, ""),
                "quali_elo_rating": round(quali_elo[d], 1) if d in quali_elo else "",
                "quali_elo_rank": quali_elo_rank.get(d, ""),
                "quali_elo_n": quali_elo_n.get(d, ""),
                "race_ridge_ms": round(race_ridge[d], 1) if d in race_ridge else "",
                "race_ridge_rank": race_ridge_rank.get(d, ""),
                "race_ridge_n": race_ridge_n.get(d, ""),
                "quali_ridge_ms": round(quali_ridge[d], 1) if d in quali_ridge else "",
                "quali_ridge_rank": quali_ridge_rank.get(d, ""),
                "quali_ridge_n": quali_ridge_n.get(d, ""),
            })
    print(f"wrote master table -> {out_path}")
