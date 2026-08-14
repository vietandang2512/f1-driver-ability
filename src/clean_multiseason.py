"""
Multi-season version of clean.py: builds pooled teammate-gap tables across
2014-2023 (not just 2023) for the ridge model, using the exact same exclusion
rules (DNF/lapped for race, best-session-reached for qualifying).

Why 2014-2023 and not 2014-present: this is meant to be the ridge analog of
"ELO's rating as of end of 2023" -- everything knowable by then, nothing from
2024-2026 leaking in. See project history for why that distinction matters.

Pooling years is also what's expected to fix ridge's disconnection problem the
same way it fixed ELO's: driver transfers across seasons link team pairs that
a single season can't.
"""

import csv
from collections import defaultdict
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

START_YEAR = 2014
END_YEAR = 2023  # inclusive -- matches the ELO end-of-2023 snapshot


def load_rows(filename):
    with open(RAW / filename, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def in_window(row):
    return row["year"] and START_YEAR <= int(row["year"]) <= END_YEAR


def best_quali_time_millis(row):
    for key in ("q3Millis", "q2Millis", "q1Millis"):
        if row[key]:
            return int(row[key])
    return None


def build_race_gap_table():
    rows = [r for r in load_rows("f1db-races-race-results.csv") if in_window(r)]

    by_team_race = defaultdict(list)
    for r in rows:
        by_team_race[(r["raceId"], r["constructorId"])].append(r)

    out_rows = []
    excluded_dnf = excluded_lapped = excluded_incomplete = 0

    for (race_id, constructor_id), pair in by_team_race.items():
        if len(pair) != 2:
            excluded_incomplete += 1
            continue
        a, b = sorted(pair, key=lambda r: r["driverId"])

        if a["reasonRetired"] or b["reasonRetired"]:
            excluded_dnf += 1
            continue
        if bool(a["gapLaps"]) or bool(b["gapLaps"]):
            excluded_lapped += 1
            continue

        a_gap = int(a["gapMillis"]) if a["gapMillis"] else 0
        b_gap = int(b["gapMillis"]) if b["gapMillis"] else 0

        out_rows.append(
            {
                "raceId": race_id,
                "year": a["year"],
                "round": a["round"],
                "constructorId": constructor_id,
                "driver_a": a["driverId"],
                "driver_b": b["driverId"],
                "signed_gap_millis": a_gap - b_gap,
            }
        )

    print(
        f"[race gap {START_YEAR}-{END_YEAR}] usable={len(out_rows)}  "
        f"excluded_dnf={excluded_dnf}  excluded_lapped={excluded_lapped}  "
        f"excluded_incomplete_pair={excluded_incomplete}"
    )
    return out_rows


def build_quali_gap_table():
    rows = [r for r in load_rows("f1db-races-qualifying-results.csv") if in_window(r)]

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

        out_rows.append(
            {
                "raceId": race_id,
                "year": a["year"],
                "round": a["round"],
                "constructorId": constructor_id,
                "driver_a": a["driverId"],
                "driver_b": b["driverId"],
                "signed_gap_millis": a_time - b_time,
            }
        )

    print(
        f"[quali gap {START_YEAR}-{END_YEAR}] usable={len(out_rows)}  "
        f"excluded_no_time={excluded_no_time}  excluded_incomplete_pair={excluded_incomplete}"
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
    write_csv(race_rows, f"race_gap_{START_YEAR}_{END_YEAR}.csv")
    write_csv(quali_rows, f"quali_gap_{START_YEAR}_{END_YEAR}.csv")
