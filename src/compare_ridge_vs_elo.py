"""
Cross-validate ridge (2014-2023 pooled) against chained ELO (snapshotted at
end of 2023) -- same metric compared across two independently-built models,
which is the real point of building both in the first place: if they agree,
that's evidence both are picking up a real signal rather than an artifact of
one particular method.

Ridge coefficients are in ms (negative = faster than average, no fixed
scale). ELO ratings are on the 1500-centered rating scale. Since the two
aren't on the same units, direct value comparison isn't meaningful --
compare RANKS and correlation instead, the same way compare_quali_vs_race_elo.py did.
"""

import csv
from pathlib import Path
from statistics import correlation

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"
SEASON = 2023


def load_ridge_coefs(filename):
    with open(DATA / filename, newline="", encoding="utf-8") as f:
        # lower (more negative) coef = faster => rank ascending by coef
        return {row["driver"]: float(row["coef_ms"]) for row in csv.DictReader(f)}


def load_elo_history(filename):
    with open(DATA / filename, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def end_of_season_snapshot(history_rows, year):
    latest = {}
    for row in history_rows:
        if int(row["year"]) != year:
            continue
        round_ = int(row["round"])
        for side in ("a", "b"):
            driver = row[f"driver_{side}"]
            rating = float(row[f"rating_{side}_after"])
            if driver not in latest or round_ > latest[driver][0]:
                latest[driver] = (round_, rating)
    return {d: r for d, (rnd, r) in latest.items()}


def rank_map(values_dict, ascending):
    ranked = sorted(values_dict.items(), key=lambda t: t[1], reverse=not ascending)
    return {driver: i + 1 for i, (driver, _) in enumerate(ranked)}


def compare(label, ridge_coefs, elo_snapshot):
    drivers = sorted(set(ridge_coefs) & set(elo_snapshot))
    print(f"\n=== {label}: {len(drivers)} drivers with both a ridge (2014-2023) "
          f"coefficient and an end-of-{SEASON} ELO rating ===")

    # IMPORTANT: rank within the same 22-driver overlap for both, not ridge's
    # full 48/56-driver 2014-2023 pool vs ELO's ~22-driver season -- ranking
    # against different-sized populations isn't a fair comparison.
    ridge_coefs_overlap = {d: ridge_coefs[d] for d in drivers}
    elo_snapshot_overlap = {d: elo_snapshot[d] for d in drivers}
    ridge_ranks = rank_map(ridge_coefs_overlap, ascending=True)   # lower ms = faster = rank 1
    elo_ranks = rank_map(elo_snapshot_overlap, ascending=False)   # higher rating = better = rank 1

    ridge_vals = [ridge_coefs[d] for d in drivers]
    elo_vals = [elo_snapshot[d] for d in drivers]
    # flip sign on ridge so "bigger = better" for both, matching correlation direction
    r = correlation([-v for v in ridge_vals], elo_vals)
    print(f"Pearson correlation (ridge coef, sign-flipped, vs ELO rating): {r:.3f}")

    rows = [(d, ridge_coefs[d], ridge_ranks[d], elo_snapshot[d], elo_ranks[d],
             ridge_ranks[d] - elo_ranks[d]) for d in drivers]

    print(f"{'driver':<20}{'ridge ms':>10}{'ridge rk':>10}{'elo':>9}{'elo rk':>8}{'rk diff':>9}")
    for d, rc, rr, ev, er, diff in sorted(rows, key=lambda t: t[2]):
        print(f"{d:<20}{rc:>10.1f}{rr:>10}{ev:>9.1f}{er:>8}{diff:>+9}")

    print(f"\nBiggest divergences (ridge rank - ELO rank):")
    for d, rc, rr, ev, er, diff in sorted(rows, key=lambda t: -abs(t[5]))[:5]:
        print(f"  {d:<20} ridge rank {rr:>2}, ELO rank {er:>2}  (diff {diff:+d})")

    return r


if __name__ == "__main__":
    race_ridge = load_ridge_coefs("ridge_race_coefs_2014_2023.csv")
    quali_ridge = load_ridge_coefs("ridge_quali_coefs_2014_2023.csv")

    race_elo_history = load_elo_history("chained_elo_history_2014_present.csv")
    quali_elo_history = load_elo_history("chained_elo_history_quali_2014_present.csv")

    race_elo_snap = end_of_season_snapshot(race_elo_history, SEASON)
    quali_elo_snap = end_of_season_snapshot(quali_elo_history, SEASON)

    r_race = compare("RACE PACE", race_ridge, race_elo_snap)
    r_quali = compare("QUALIFYING PACE", quali_ridge, quali_elo_snap)

    print(f"\n=== Summary ===")
    print(f"Race:  ridge (2014-2023) vs ELO (end of {SEASON}) correlation = {r_race:.3f}")
    print(f"Quali: ridge (2014-2023) vs ELO (end of {SEASON}) correlation = {r_quali:.3f}")
