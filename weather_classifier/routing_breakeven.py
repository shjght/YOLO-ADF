# -*- coding: utf-8 -*-
"""Break-even routing-accuracy analysis (needs NO classifier - Table 3 only).

Answers: how accurate must scene-level routing be before weather switching
beats (a) the baseline, (b) the best single fixed model (ADF-N)?

Model: with probability p the scene is routed to its oracle branch (the WADI
choice); with probability 1-p it is misrouted, either to the average of the
other routable branches (expected case) or to the worst one (adversarial).
WADI_real(p) = p * oracle + (1-p) * wrong, per scene, then averaged.

    python routing_breakeven.py [--table3 table3.csv]
"""
import argparse, csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENES = ["fog", "night", "rain", "snow"]
ORACLE = {"fog": "ADF-F", "night": "ADF-N", "rain": "CA", "snow": "ADF-S"}
ROUTABLE = ["ADF-F", "ADF-S", "ADF-N", "CA"]   # conservative set (AMAR excluded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table3", default=str(HERE / "table3.csv"))
    args = ap.parse_args()
    t3 = {}
    with open(args.table3, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t3[row["model"].strip()] = {s: float(row[s]) for s in SCENES}

    baseline = sum(t3["Baseline"][s] for s in SCENES) / 4
    best_single = max(sum(t3[m][s] for s in SCENES) / 4 for m in t3)
    wadi_oracle = sum(t3[ORACLE[s]][s] for s in SCENES) / 4

    print(f"baseline SSDI      = {baseline:.4f}")
    print(f"best single model  = {best_single:.4f}")
    print(f"WADI oracle bound  = {wadi_oracle:.4f}\n")

    for mode, pick in [("expected (wrong = avg of others)", lambda w, o: sum(t3[b][w] for b in ROUTABLE if b != o) / (len(ROUTABLE) - 1)),
                       ("adversarial (wrong = worst)", lambda w, o: min(t3[b][w] for b in ROUTABLE if b != o))]:
        print(f"--- misroute model: {mode} ---")
        print(f"{'scene':6s} {'oracle':>7s} {'wrong':>7s} {'p: >baseline':>13s} {'best-fixed':>11s}")
        for w in SCENES:
            o = ORACLE[w]
            right, wrong = t3[o][w], pick(w, o)
            best_fixed_on_scene = max(t3[m][w] for m in t3)
            def p_needed(ref):
                if right == wrong:
                    return 0.0 if wrong >= ref else float('inf')
                p = (ref - wrong) / (right - wrong)
                return max(p, 0.0)
            print(f"{w:6s} {right:7.4f} {wrong:7.4f} {p_needed(t3['Baseline'][w]):13.1%} {best_fixed_on_scene:11.4f}")
        wrong_avg_wadi = sum(pick(w, ORACLE[w]) for w in SCENES) / 4
        slope = wadi_oracle - wrong_avg_wadi
        def p_wadi(ref):
            if slope <= 0:
                return 0.0 if wrong_avg_wadi >= ref else float('inf')
            return max((ref - wrong_avg_wadi) / slope, 0.0)
        print(f"WADI(p) = {wrong_avg_wadi:.4f} + {slope:.4f}*p")
        print(f"  switching beats baseline     at p > {p_wadi(baseline):.1%}")
        print(f"  switching beats best single  at p > {p_wadi(best_single):.1%}")
        for r in (0.7, 0.8, 0.9):
            target = baseline + r * (wadi_oracle - baseline)
            print(f"  retains {r:.0%} of the WADI-SSDI gain at p > {p_wadi(target):.1%}")
        print()


if __name__ == "__main__":
    main()
