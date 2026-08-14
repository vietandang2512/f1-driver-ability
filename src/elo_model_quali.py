"""
Qualifying-pace counterpart to elo_model.py.

Same chained-ELO mechanism (teammate win/loss, K=24, 2014-present, chronological),
applied to qualifying classification instead of race classification. Deliberately
NOT restricted to 2023 -- a single season can't chain (see project history: this is
exactly what broke the 2023-only ridge regression). Restricted comparisons for a
given season are done afterward by snapshotting the rating history, not by limiting
the input data.

Note this is simpler than the race version in one respect: qualifying's `gap`
column in F1DB is unreliable (only populated for Q3 participants -- see clean.py's
notes), but `positionNumber` -- the final, official qualifying classification -- is
reliable and exactly what a win/loss signal needs, so no q1/q2/q3 reconstruction is
required here.
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


def run_chained_elo():
    results = [
        r
        for r in load_rows("f1db-races-qualifying-results.csv")
        if r["year"] and int(r["year"]) >= ERA_START_YEAR
    ]

    by_race = defaultdict(list)
    for r in results:
        by_race[(int(r["year"]), int(r["round"]), r["raceId"])].append(r)

    race_keys = sorted(by_race.keys())

    ratings = defaultdict(lambda: INITIAL_RATING)
    races_seen = defaultdict(int)
    history = []

    comparisons_used = 0
    comparisons_skipped_no_pair = 0

    for year, round_, race_id in race_keys:
        rows = by_race[(year, round_, race_id)]
        by_constructor = defaultdict(list)
        for r in rows:
            by_constructor[r["constructorId"]].append(r)

        for constructor_id, pair in by_constructor.items():
            classified = [r for r in pair if r["positionNumber"]]
            if len(classified) != 2:
                comparisons_skipped_no_pair += 1
                continue

            a, b = sorted(classified, key=lambda r: r["driverId"])
            pos_a, pos_b = int(a["positionNumber"]), int(b["positionNumber"])

            ra, rb = ratings[a["driverId"]], ratings[b["driverId"]]
            exp_a = expected_score(ra, rb)
            actual_a = 1.0 if pos_a < pos_b else 0.0

            new_ra = ra + K_FACTOR * (actual_a - exp_a)
            new_rb = rb + K_FACTOR * ((1 - actual_a) - (1 - exp_a))

            ratings[a["driverId"]] = new_ra
            ratings[b["driverId"]] = new_rb
            races_seen[a["driverId"]] += 1
            races_seen[b["driverId"]] += 1
            comparisons_used += 1

            history.append(
                {
                    "year": year,
                    "round": round_,
                    "raceId": race_id,
                    "constructorId": constructor_id,
                    "driver_a": a["driverId"],
                    "driver_b": b["driverId"],
                    "winner": a["driverId"] if actual_a == 1.0 else b["driverId"],
                    "rating_a_after": round(new_ra, 1),
                    "rating_b_after": round(new_rb, 1),
                }
            )

    print(
        f"Processed {len(race_keys)} qualifying sessions from {ERA_START_YEAR}-present; "
        f"{comparisons_used} teammate comparisons used, "
        f"{comparisons_skipped_no_pair} team-sessions skipped (no valid pair)."
    )
    return ratings, races_seen, history


def connected_components(history_rows):
    adj = defaultdict(set)
    for r in history_rows:
        adj[r["driver_a"]].add(r["driver_b"])
        adj[r["driver_b"]].add(r["driver_a"])
    seen, comps = set(), []
    for node in adj:
        if node in seen:
            continue
        stack, comp = [node], set()
        while stack:
            n = stack.pop()
            if n in comp:
                continue
            comp.add(n)
            stack.extend(adj[n] - comp)
        seen |= comp
        comps.append(comp)
    return comps


def write_history(history_rows):
    path = OUT / "chained_elo_history_quali_2014_present.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)
    print(f"wrote {len(history_rows)} rows -> {path}")


if __name__ == "__main__":
    ratings, races_seen, history = run_chained_elo()

    comps = connected_components(history)
    print(f"\nConnected components in the 2014-present qualifying teammate graph: {len(comps)}")
    for c in sorted(comps, key=len, reverse=True)[:5]:
        print(f"  size {len(c)}: {sorted(c)[:6]}{' ...' if len(c) > 6 else ''}")

    ranked = sorted(ratings.items(), key=lambda t: t[1], reverse=True)
    print(f"\n=== CHAINED QUALIFYING ELO RANKING, {ERA_START_YEAR}-present ({len(ranked)} drivers) ===")
    print(f"{'rank':<5}{'driver':<20}{'rating':>8}{'races':>8}")
    for i, (driver, rating) in enumerate(ranked, start=1):
        print(f"{i:<5}{driver:<20}{rating:>8.1f}{races_seen[driver]:>8}")

    write_history(history)
