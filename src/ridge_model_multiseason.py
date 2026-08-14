"""
Ridge regression on the pooled 2014-2023 teammate-gap tables (see
clean_multiseason.py). Same fitting logic as ridge_model.py, reused here --
the only thing that changed is how many rows get fed in. Checks connectivity
the same way elo_model.py did, to see whether pooling years fixes ridge's
disconnection problem the way it fixed ELO's.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"
START_YEAR, END_YEAR = 2014, 2023


def load_pairs(filename):
    with open(DATA / filename, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def connected_components(rows):
    adj = defaultdict(set)
    for r in rows:
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

    n_pairings = {d: 0 for d in drivers}
    for r in rows:
        n_pairings[r["driver_a"]] += 1
        n_pairings[r["driver_b"]] += 1

    results = sorted(
        zip(drivers, model.coef_, [n_pairings[d] for d in drivers]),
        key=lambda t: t[1],
    )
    return results, model.alpha_


def print_ranking(title, results, chosen_alpha):
    print(f"\n=== {title} (ridge alpha={chosen_alpha:g}) ===")
    print(f"{'driver':<20}{'coef (ms)':>10}{'n races':>9}")
    for driver, coef, n in results:
        print(f"{driver:<20}{coef:>10.1f}{n:>9}")


def write_coefs(results, filename):
    path = DATA / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["driver", "coef_ms", "n_pairings"])
        for driver, coef, n in results:
            writer.writerow([driver, round(coef, 2), n])
    print(f"wrote {len(results)} rows -> {path}")


if __name__ == "__main__":
    for label, filename, out_name in [
        ("RACE", f"race_gap_{START_YEAR}_{END_YEAR}.csv", f"ridge_race_coefs_{START_YEAR}_{END_YEAR}.csv"),
        ("QUALI", f"quali_gap_{START_YEAR}_{END_YEAR}.csv", f"ridge_quali_coefs_{START_YEAR}_{END_YEAR}.csv"),
    ]:
        rows = load_pairs(filename)
        comps = connected_components(rows)
        print(f"\n{label}: {len(rows)} rows, {len(comps)} connected components "
              f"({sum(len(c) for c in comps)} drivers total)")
        for c in sorted(comps, key=len, reverse=True)[:5]:
            print(f"  size {len(c)}: {sorted(c)[:6]}{' ...' if len(c) > 6 else ''}")

        results, alpha = fit_pairwise_ridge(rows)
        print_ranking(f"{label} PACE, {START_YEAR}-{END_YEAR} pooled ridge", results, alpha)
        write_coefs(results, out_name)
