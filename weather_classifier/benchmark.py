# -*- coding: utf-8 -*-
"""Step 3: paper-facing benchmark - accuracy (BDD val in-domain + ACDC val
cross-domain) and routing latency.

    python benchmark.py --model mobilenet_v3_small
    python benchmark.py --model mobilenet_v3_small --cpu-latency
    python benchmark.py --model resnet18 --acdc-root C:/Users/KUST/xl/acdc

Latency protocol (defensible in review): batch=1, 224x224, 100 warmup iters,
500 timed iters, CUDA events with torch.cuda.synchronize; FP32 and FP16;
CPU latency with default torch threads. FLOPs via thop if installed.

Outputs results/benchmark_<model>.json plus a ready-to-paste markdown table.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import CLASSES, find_acdc_root, find_data_root, set_seed
from dataset import AcdcValDataset, BddWeatherDataset
from models import build_model, count_params


@torch.no_grad()
def accuracy_per_class(model, loader, device, amp=True):
    """Returns overall acc, macro-F1, per-class dict, total, and a confusion
    matrix (rows = true class, cols = predicted class, order = CLASSES).
    The confusion matrix is what compute_wadi_real.py uses for scene-level
    routing simulation."""
    model.eval()
    tp = {c: 0 for c in CLASSES}
    fp = {c: 0 for c in CLASSES}
    fn = {c: 0 for c in CLASSES}
    nc = len(CLASSES)
    cm = torch.zeros(nc, nc, dtype=torch.long)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
            pred = model(x).argmax(1).cpu()
        y = y.cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
        cm.index_put_((y, pred), torch.ones(len(y), dtype=torch.long), accumulate=True)
        for c_idx, c in enumerate(CLASSES):
            match = (pred == c_idx) & (y == c_idx)
            tp[c] += match.sum().item()
            fp[c] += ((pred == c_idx) & (y != c_idx)).sum().item()
            fn[c] += ((pred != c_idx) & (y == c_idx)).sum().item()
    per_class = {}
    f1s = []
    for c in CLASSES:
        prec = tp[c] / max(tp[c] + fp[c], 1)
        rec = tp[c] / max(tp[c] + fn[c], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        n = tp[c] + fn[c]
        per_class[c] = {"n": n, "precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4)}
        if n > 0:
            f1s.append(f1)
    return correct / max(total, 1), sum(f1s) / max(len(f1s), 1), per_class, total, cm


@torch.no_grad()
def gpu_latency(model, device, size, iters=500, warmup=100, fp16=False):
    """Mean/std ms per image, batch=1, CUDA-event timed."""
    x = torch.randn(1, 3, size, size, device=device)
    if fp16:
        model = model.half()
        x = x.half()
    # warmup
    for _ in range(warmup):
        with torch.autocast(device_type="cuda", enabled=fp16):
            model(x)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    times = []
    for _ in range(iters):
        start.record()
        with torch.autocast(device_type="cuda", enabled=fp16):
            model(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


@torch.no_grad()
def cpu_latency(model, size, iters=200, warmup=20):
    x = torch.randn(1, 3, size, size)
    model.eval()
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        times.append((time.perf_counter() - t0) * 1000)
    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def flops(model, size):
    try:
        from thop import profile
        x = torch.randn(1, 3, size, size)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return round(macs * 2 / 1e9, 3)  # report GMACs*2 ~ GFLOPs convention
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenet_v3_small")
    ap.add_argument("--weights", default=None, help="default: weights/weatherclf_<model>_best.pt")
    ap.add_argument("--csv", default=str(Path(__file__).resolve().parent / "labels_bdd.csv"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--acdc-root", default=None)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cpu-latency", action="store_true",
                    help="also measure CPU latency (slow; for the desktop-vs-automotive note)")
    args = ap.parse_args()

    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    here = Path(__file__).resolve().parent
    res_dir = here / "results"
    res_dir.mkdir(exist_ok=True)

    ckpt_path = Path(args.weights) if args.weights else \
        here / "weights" / f"weatherclf_{args.model}_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt.get("model", args.model), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    size = ckpt.get("size", args.size)
    print(f"loaded {ckpt_path.name}  (val_acc at train time {ckpt.get('val_acc')})")

    out = {"model": ckpt.get("model", args.model), "params_M": round(count_params(model), 3),
           "input_size": size, "device": str(device)}

    # ---- accuracy: BDD val (in-domain) ----
    if Path(args.csv).exists():
        ds = BddWeatherDataset(args.csv, split="val", size=size)
        dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers)
        acc, mf1, per, n, cm = accuracy_per_class(model, dl, device)
        out["bdd_val"] = {"n": n, "acc": round(acc, 4), "macro_f1": round(mf1, 4),
                          "per_class": per, "confusion_matrix": cm.tolist()}
        print(f"\n[BDD val, in-domain]  n={n}  acc={acc:.4f}  macro-F1={mf1:.4f}")
        for c, v in per.items():
            print(f"  {c:6s} n={v['n']:5d}  P={v['precision']:.3f}  "
                  f"R={v['recall']:.3f}  F1={v['f1']:.3f}")
    else:
        print(f"[skip] {args.csv} not found (BDD val accuracy skipped)")

    # ---- accuracy: ACDC val (cross-domain, adverse-only) ----
    try:
        acdc_root = Path(args.acdc_root) if args.acdc_root else \
            find_acdc_root(Path(args.data_root) if args.data_root else find_data_root())
        ds = AcdcValDataset(acdc_root, size=size)
        dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers)
        acc, mf1, per, n, cm = accuracy_per_class(model, dl, device)
        # per-image predictions (order preserved: shuffle=False) - enables
        # scene-level bootstrap CIs for realized WADI in compute_wadi_real.py
        per_image = []
        samples = ds.samples
        with torch.no_grad():
            i0 = 0
            for x, _ in dl:
                chunk = samples[i0:i0 + x.size(0)]
                i0 += x.size(0)
                with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                    p = model(x.to(device)).argmax(1).cpu().tolist()
                for (path, y), pi in zip(chunk, p):
                    per_image.append({"path": path, "true": int(y), "pred": int(pi)})
            assert i0 == len(samples), f"per-image pass covered {i0}/{len(samples)}"
        out["acdc_val"] = {"root": str(acdc_root), "n": n, "acc": round(acc, 4),
                           "macro_f1": round(mf1, 4), "per_class": per,
                           "confusion_matrix": cm.tolist(), "per_image": per_image}
        print(f"\n[ACDC val, cross-domain]  n={n}  acc={acc:.4f}  macro-F1={mf1:.4f}")
        for c, v in per.items():
            if v["n"]:
                print(f"  {c:6s} n={v['n']:5d}  P={v['precision']:.3f}  "
                      f"R={v['recall']:.3f}  F1={v['f1']:.3f}")
    except FileNotFoundError as e:
        print(f"[skip] ACDC eval: {e}")

    # ---- latency ----
    if device.type == "cuda":
        ms32, sd32 = gpu_latency(model, device, size, fp16=False)
        ms16, sd16 = gpu_latency(model, device, size, fp16=True)
        out["latency_gpu"] = {"batch": 1, "fp32_ms": round(ms32, 3),
                              "fp32_std": round(sd32, 3),
                              "fp32_fps": round(1000 / ms32, 1),
                              "fp16_ms": round(ms16, 3), "fp16_std": round(sd16, 3),
                              "fp16_fps": round(1000 / ms16, 1)}
        print(f"\n[latency GPU]  FP32 {ms32:.3f}+-{sd32:.3f} ms "
              f"({1000/ms32:.0f} FPS)   FP16 {ms16:.3f}+-{sd16:.3f} ms "
              f"({1000/ms16:.0f} FPS)")
    else:
        print("[latency] CUDA not available, GPU latency skipped")

    g = flops(model, size)
    if g:
        out["flops_G"] = g
        print(f"[FLOPs] {g} G (thop)")

    if args.cpu_latency:
        ms, sd = cpu_latency(model, size)
        out["latency_cpu"] = {"ms": round(ms, 2), "std": round(sd, 2),
                              "fps": round(1000 / ms, 1)}
        print(f"[latency CPU] {ms:.2f}+-{sd:.2f} ms ({1000/ms:.1f} FPS)")

    out_path = res_dir / f"benchmark_{args.model}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # markdown snippet for the manuscript
    lines = ["| Model | Params (M) | FLOPs (G) | BDD val Acc | ACDC val Acc | "
             "GPU FP32 (ms) | GPU FP16 (ms) |",
             "|---|---|---|---|---|---|---|"]
    b = out.get("bdd_val", {}); a = out.get("acdc_val", {})
    lat = out.get("latency_gpu", {})
    lines.append(f"| {out['model']} | {out['params_M']} | {out.get('flops_G', '-')} | "
                 f"{b.get('acc', '-')} | {a.get('acc', '-')} | "
                 f"{lat.get('fp32_ms', '-')} | {lat.get('fp16_ms', '-')} |")
    md = res_dir / f"table_{args.model}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsaved {out_path}\nsaved {md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
