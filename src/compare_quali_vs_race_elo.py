"""
Compare qualifying-chained-ELO vs race-chained-ELO for the 2023 season specifically.

Both ELO chains run across the full 2014-present window (required for chaining to
work at all -- see project history), so a fair season-specific comparison means
snapshotting each driver's rating as of their LAST 2023 race, not their final
(2026-inclusive) rating. This script does that snapshotting, then compares.
"""

import csv
from pathlib import Path
from statistics import correlation

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"
SEASON = 2023


def load_history(filename):
    with open(DATA / filename, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def end_of_season_snapshot(history_rows, year):
    """For each driver, the rating right after their last comparison in `year`."""
    # (round, rating_after) per driver, keep the max round
    latest = {}  # driverId -> (round, rating)
    for row in history_rows:
        if int(row["year"]) != year:
            continue
        round_ = int(row["round"])
        for side, other_side in (("a", "b"), ("b", "a")):
            driver = row[f"driver_{side}"]
            rating = float(row[f"rating_{side}_after"])
            if driver not in latest or round_ > latest[driver][0]:
                latest[driver] = (round_, rating)
    return {d: r for d, (rnd, r) in latest.items()}


def rank_map(ratings_dict):
    ranked = sorted(ratings_dict.items(), key=lambda t: t[1], reverse=True)
    return {driver: i + 1 for i, (driver, _) in enumerate(ranked)}


if __name__ == "__main__":
    race_history = load_history("chained_elo_history_2014_present.csv")
    quali_history = load_history("chained_elo_history_quali_2014_present.csv")

    race_snap = end_of_season_snapshot(race_history, SEASON)
    quali_snap = end_of_season_snapshot(quali_history, SEASON)

    drivers = sorted(set(race_snap) & set(quali_snap))
    print(f"Drivers with an end-of-{SEASON} snapshot in BOTH chains: {len(drivers)}")
    only_race = set(race_snap) - set(quali_snap)
    only_quali = set(quali_snap) - set(race_snap)
    if only_race:
        print(f"  in race only: {sorted(only_race)}")
    if only_quali:
        print(f"  in quali only: {sorted(only_quali)}")

    race_ranks = rank_map(race_snap)
    quali_ranks = rank_map(quali_snap)

    rows = []
    for d in drivers:
        rr, qr = race_ranks[d], quali_ranks[d]
        rows.append((d, race_snap[d], rr, quali_snap[d], qr, rr - qr))

    # overall agreement
    race_vals = [race_snap[d] for d in drivers]
    quali_vals = [quali_snap[d] for d in drivers]
    r = correlation(race_vals, quali_vals)
    print(f"\nPearson correlation between race-ELO and quali-ELO (end of {SEASON}): {r:.3f}")

    print(f"\n{'driver':<20}{'race elo':>10}{'race rk':>9}{'quali elo':>11}{'quali rk':>10}{'rk diff':>9}")
    for d, rv, rr, qv, qr, diff in sorted(rows, key=lambda t: t[2]):
        print(f"{d:<20}{rv:>10.1f}{rr:>9}{qv:>11.1f}{qr:>10}{diff:>+9}")

    print(f"\n=== Biggest divergences (positive = much better at RACE than QUALIFYING) ===")
    for d, rv, rr, qv, qr, diff in sorted(rows, key=lambda t: -t[5])[:5]:
        print(f"  {d:<20} race rank {rr:>2}, quali rank {qr:>2}  (race - quali rank = {diff:+d})")
    print(f"\n=== Biggest divergences (negative = much better at QUALIFYING than RACE) ===")
    for d, rv, rr, qv, qr, diff in sorted(rows, key=lambda t: t[5])[:5]:
        print(f"  {d:<20} race rank {rr:>2}, quali rank {qr:>2}  (race - quali rank = {diff:+d})")
