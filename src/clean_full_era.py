"""
Same as clean_multiseason.py but with no upper year bound -- pools the full
2014-present window (matching what elo_model.py / elo_model_quali.py already
cover), for the ridge models. This is now the "final" ridge scope; the
2014-2023 version stays in the repo since it's what the ridge-vs-ELO
end-of-2023 comparison legitimately needed (no future-data leakage).
"""

import csv
from collections import defaultdict
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

START_YEAR = 2014


def load_rows(filename):
    with open(RAW / filename, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def in_window(row):
    return row["year"] and int(row["year"]) >= START_YEAR


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
                "raceId": race_id, "year": a["year"], "round": a["round"],
                "constructorId": constructor_id,
                "driver_a": a["driverId"], "driver_b": b["driverId"],
                "signed_gap_millis": a_gap - b_gap,
            }
        )
    print(f"[race gap {START_YEAR}-present] usable={len(out_rows)}  "
          f"excluded_dnf={excluded_dnf}  excluded_lapped={excluded_lapped}  "
          f"excluded_incomplete_pair={excluded_incomplete}")
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
                "raceId": race_id, "year": a["year"], "round": a["round"],
                "constructorId": constructor_id,
                "driver_a": a["driverId"], "driver_b": b["driverId"],
                "signed_gap_millis": a_time - b_time,
            }
        )
    print(f"[quali gap {START_YEAR}-present] usable={len(out_rows)}  "
          f"excluded_no_time={excluded_no_time}  excluded_incomplete_pair={excluded_incomplete}")
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
    write_csv(build_race_gap_table(), f"race_gap_{START_YEAR}_present.csv")
    write_csv(build_quali_gap_table(), f"quali_gap_{START_YEAR}_present.csv")
