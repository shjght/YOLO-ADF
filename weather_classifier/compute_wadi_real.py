# -*- coding: utf-8 -*-
"""Realized WADI under classifier routing (run AFTER training the classifier).

Deployed pipeline: frame -> classifier -> pick fusion branch -> detector.
WADI in the paper is the ORACLE bound (true weather known). This script
computes what the deployed system would actually achieve, using:
  1. the classifier's confusion matrix on ACDC val (from benchmark.py), and
  2. Table 3 (per-scene mAP50 of every variant) in table3.csv.

Scene-level routing: weather is persistent, so a scene is routed by the
branch most often predicted for it. For true scene w the majority predicted
class p* selects branch map[p*]; realized mAP50(w) = Table3[map[p*], w].

    python compute_wadi_real.py --benchmark results/benchmark_mobilenet_v3_small.json

table3.csv columns: model,fog,night,rain,snow  (one row per variant,
mAP50 values copied from the paper's Table 3 - fill once, reuse).
"""
import argparse, csv, json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# predicted class -> deployed branch/model (matches the WADI routing table)
ROUTING = {"clear": "AMAR",     # in-domain best under normal weather
           "rain":  "CA",       # best in rain
           "snow":  "ADF-S",
           "fog":   "ADF-F",
           "night": "ADF-N"}
# Conservative routing: never routes to AMAR (the variant with catastrophic
# cross-domain rain/snow scores). 'clear' falls back to the balanced ADF-N.
# In-domain cost ~0.3 pp; removes the only catastrophic misroute path
# (adverse scene -> AMAR). floor(WADI_real) with this map ~= baseline SSDI.
ROUTING_SAFE = {"clear": "ADF-N",
                "rain":  "CA",
                "snow":  "ADF-S",
                "fog":   "ADF-F",
                "night": "ADF-N"}
SCENES = ["fog", "night", "rain", "snow"]
WADI_ORACLE = {"fog": "ADF-F", "night": "ADF-N", "rain": "CA", "snow": "ADF-S"}


def load_table3(path):
    t3 = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t3[row["model"].strip()] = {s: float(row[s]) for s in SCENES}
    return t3


CLASSES5 = ["clear", "rain", "snow", "fog", "night"]


def bootstrap_wadi(per_image, routing, t3, B=1000, seed=0):
    """95% CI for realized WADI via scene-level bootstrap: resample each
    scene's ACDC val images with replacement, re-take the majority vote,
    re-route, re-look-up Table 3. Needs the per_image list benchmark.py saves
    (returns None when absent, e.g. for older benchmark jsons)."""
    if not per_image:
        return None
    by_scene = {w: [] for w in SCENES}
    for rec in per_image:
        w = CLASSES5[rec["true"]]
        if w in by_scene:
            by_scene[w].append(rec["pred"])
    scene_lists = [(w, by_scene[w]) for w in SCENES if by_scene[w]]
    if any(len(p) < 30 for _, p in scene_lists):
        print(f"[bootstrap] skipped: a scene has <30 images "
              f"({ {w: len(p) for w, p in scene_lists} })")
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        accs = []
        for w, preds in scene_lists:
            counts = Counter(rng.choice(preds) for _ in preds)
            p_star = counts.most_common(1)[0][0]
            accs.append(t3[routing[CLASSES5[p_star]]][w])
        draws.append(sum(accs) / len(accs))
    draws.sort()
    mean = sum(draws) / len(draws)
    lo, hi = draws[int(0.025 * B)], draws[int(0.975 * B) - 1]
    return mean, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=str(HERE / "results" / "benchmark_mobilenet_v3_small.json"))
    ap.add_argument("--table3", default=str(HERE / "table3.csv"))
    args = ap.parse_args()

    bench = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    t3 = load_table3(args.table3)
    cm = bench["acdc_val"]["confusion_matrix"]     # rows=true(5 classes), cols=pred
    classes = ["clear", "rain", "snow", "fog", "night"]

    print(f"ACDC val n={bench['acdc_val']['n']}  acc={bench['acdc_val']['acc']}\n")
    summary = {}
    for rname, routing in [("aggressive (clear->AMAR)", ROUTING),
                           ("conservative (clear->ADF-N)", ROUTING_SAFE)]:
        print(f"===== routing: {rname} =====")
        realized, rows = [], []
        for w in SCENES:
            ti = classes.index(w)
            counts = {classes[c]: cm[ti][c] for c in range(len(classes))}
            pred_star = max(counts, key=counts.get)
            branch = routing[pred_star]
            acc_w = t3[branch][w]
            oracle_branch = WADI_ORACLE[w]
            oracle_w = t3[oracle_branch][w]
            realized.append(acc_w)
            rows.append((w, pred_star, sum(counts.values()), branch, acc_w, oracle_branch, oracle_w))
            dist = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
            print(f"[{w:5s}] predicted -> {dist}")
            print(f"        routed to {branch:6s} mAP50={acc_w:.4f}   "
                  f"(oracle: {oracle_branch} {oracle_w:.4f}, delta {acc_w-oracle_w:+.4f})")
        wadi_real = sum(realized) / len(realized)
        wadi_oracle = sum(r[6] for r in rows) / len(rows)
        summary[rname] = {"wadi_oracle": round(wadi_oracle, 4),
                          "wadi_realized": round(wadi_real, 4),
                          "retention_pct": round(100 * wadi_real / wadi_oracle, 1),
                          "per_scene": [{"scene": r[0], "majority_pred": r[1], "n": r[2],
                                         "routed_branch": r[3], "realized_map50": r[4],
                                         "oracle_branch": r[5], "oracle_map50": r[6]} for r in rows]}
        ci = bootstrap_wadi(bench["acdc_val"].get("per_image"), routing, t3)
        if ci:
            mean, lo, hi = ci
            summary[rname]["wadi_realized_ci95"] = [round(lo, 4), round(hi, 4)]
            print(f"WADI (oracle)   = {wadi_oracle:.4f}")
            print(f"WADI (realized) = {wadi_real:.4f}   "
                  f"bootstrap95% [{lo:.4f}, {hi:.4f}]   "
                  f"retention {100*wadi_real/wadi_oracle:.1f}%\n")
        else:
            print(f"WADI (oracle)   = {wadi_oracle:.4f}")
            print(f"WADI (realized) = {wadi_real:.4f}   "
                  f"retention {100*wadi_real/wadi_oracle:.1f}%\n")
    out = HERE / "results" / "wadi_real.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
