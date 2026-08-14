"""
Build the two teammate-gap tables (qualifying and race) that feed ridge_model.py.

Rules locked in during project planning (see project doc for full rationale):
  - Season: 2023 (MVP season — real mid-season driver swaps, full OpenF1 overlap).
  - Race gap target: signed gapMillis difference between teammates at the flag.
    Excludes a race/team-pair if either teammate DNF'd OR was lapped (gapMillis
    is only populated for lead-lap cars in F1DB, so a lapped car has no usable
    millisecond gap).
  - Qualifying gap target: signed difference between teammates' best session
    time (max of q1/q2/q3 reached), NOT the precomputed `gap` column, which
    F1DB only populates for Q3 participants.
  - Sign convention: gap is DRIVER_A minus DRIVER_B where (A, B) is the
    alphabetically-first-then-second driverId pair for that constructor in
    that race, in milliseconds. Positive means A was SLOWER / lost time to B.
    Kept relative to a canonical order (not "car number 1 vs 2") so each
    driver's rows can be built consistently regardless of who's "first".
"""

import csv
from collections import defaultdict
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

SEASON = "2023"


def load_rows(filename):
    with open(RAW / filename, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_quali_time_millis(row):
    """Best (fastest) session time the driver actually set, deepest session first."""
    for key in ("q3Millis", "q2Millis", "q1Millis"):
        if row[key]:
            return int(row[key])
    return None


def build_race_gap_table():
    rows = [r for r in load_rows("f1db-races-race-results.csv") if r["year"] == SEASON]

    by_team_race = defaultdict(list)
    for r in rows:
        by_team_race[(r["raceId"], r["constructorId"])].append(r)

    out_rows = []
    excluded_dnf = excluded_lapped = excluded_incomplete = 0

    for (race_id, constructor_id), pair in by_team_race.items():
        if len(pair) != 2:
            excluded_incomplete += 1
            continue
        a, b = sorted(pair, key=lambda r: r["driverId"])  # canonical order

        if a["reasonRetired"] or b["reasonRetired"]:
            excluded_dnf += 1
            continue

        a_lapped = bool(a["gapLaps"])
        b_lapped = bool(b["gapLaps"])
        if a_lapped or b_lapped:
            excluded_lapped += 1
            continue

        # gapMillis is blank for the race leader (gap = 0 by definition)
        a_gap = int(a["gapMillis"]) if a["gapMillis"] else 0
        b_gap = int(b["gapMillis"]) if b["gapMillis"] else 0
        signed_gap = a_gap - b_gap  # positive => driver_a slower than driver_b

        out_rows.append(
            {
                "raceId": race_id,
                "round": a["round"],
                "constructorId": constructor_id,
                "driver_a": a["driverId"],
                "driver_b": b["driverId"],
                "signed_gap_millis": signed_gap,
            }
        )

    print(
        f"[race gap] usable={len(out_rows)}  excluded_dnf={excluded_dnf}  "
        f"excluded_lapped={excluded_lapped}  excluded_incomplete_pair={excluded_incomplete}"
    )
    return out_rows


def build_quali_gap_table():
    rows = [r for r in load_rows("f1db-races-qualifying-results.csv") if r["year"] == SEASON]

    by_team_race = defaultdict(list)
    for r in rows:
        by_team_race[(r["raceId"], r["constructorId"])].append(r)

    out_rows = []
    excluded_no_time = excluded_incomplete = 0

    for (race_id, constructor_id), pair in by_team_race.items():
        if len(pair) != 2:
            excluded_incomplete += 1
            continue
        a, b = sorted(pair, key=lambda r: r["driverId"])

        a_time = best_quali_time_millis(a)
        b_time = best_quali_time_millis(b)
        if a_time is None or b_time is None:
            excluded_no_time += 1
            continue

        signed_gap = a_time - b_time  # positive => driver_a slower than driver_b

        out_rows.append(
            {
                "raceId": race_id,
                "round": a["round"],
                "constructorId": constructor_id,
                "driver_a": a["driverId"],
                "driver_b": b["driverId"],
                "signed_gap_millis": signed_gap,
            }
        )

    print(
        f"[quali gap] usable={len(out_rows)}  excluded_no_time={excluded_no_time}  "
        f"excluded_incomplete_pair={excluded_incomplete}"
    )
    return out_rows


def write_csv(rows, filename):
    if not rows:
        return
    path = OUT / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    race_rows = build_race_gap_table()
    quali_rows = build_quali_gap_table()
    write_csv(race_rows, "race_gap_2023.csv")
    write_csv(quali_rows, "quali_gap_2023.csv")
