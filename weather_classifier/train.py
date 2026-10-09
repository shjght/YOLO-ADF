# -*- coding: utf-8 -*-
"""Step 2: train the weather classifier on BDD100K.

Single GPU is enough (~63k images, tiny backbones, 10 epochs < 1 h on a 3090).
    python train.py --model mobilenet_v3_small
    python train.py --model resnet18 --epochs 15
On the training machine pick a card first, e.g.:
    set CUDA_VISIBLE_DEVICES=0 && python train.py --model mobilenet_v3_small
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import CLASSES, find_data_root, set_seed
from dataset import BddWeatherDataset, class_weights
from models import build_model, count_params


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenet_v3_small", help="see models.py BACKBONES")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--csv", default=str(Path(__file__).resolve().parent / "labels_bdd.csv"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="paper convention: seed 0")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    here = Path(__file__).resolve().parent
    out_dir = here / "weights"
    out_dir.mkdir(exist_ok=True)

    if not Path(args.csv).exists():
        raise SystemExit(f"{args.csv} not found - run build_labels.py first "
                         f"(or pass --csv).")

    if args.data_root:
        # csv stores absolute paths from build time; if the dataset moved,
        # rebuild the csv on this machine instead of passing --data-root here.
        print(f"[info] note: --data-root is not used at train time; if image paths in "
              f"the csv are stale, rerun build_labels.py --data-root {args.data_root}")

    ds_tr = BddWeatherDataset(args.csv, split="train", size=args.size, train=True)
    ds_va = BddWeatherDataset(args.csv, split="val", size=args.size, train=False)
    print(f"train {len(ds_tr)}  val {len(ds_va)}  device {device}  model {args.model}")
    if len(ds_tr) == 0:
        raise SystemExit("empty train split - rebuild labels_bdd.csv on the machine "
                         "that holds the training images (build_labels.py).")

    labels_tr = [y for _, y in ds_tr.samples]
    w, counts = class_weights(labels_tr)
    print("class counts:", dict(zip(CLASSES, counts.tolist())))
    print("class weights:", [round(x, 3) for x in w.tolist()])

    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    model = build_model(args.model).to(device)
    print(f"params: {count_params(model):.2f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(weight=w.to(device))
    amp_on = bool(args.amp) and device.type == "cuda"
    try:  # torch >= 2.3 moved GradScaler into torch.amp
        scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_on)

    best = 0.0
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=scaler.is_enabled()):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * y.numel()
            seen += y.numel()
        sched.step()
        va = evaluate(model, dl_va, device)
        history.append({"epoch": ep, "loss": run_loss / seen, "val_acc": va})
        star = ""
        if va > best:
            best = va
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "classes": CLASSES, "val_acc": va, "size": args.size,
                        "epoch": ep, "seed": args.seed},
                       out_dir / f"weatherclf_{args.model}_best.pt")
            star = "  *best*"
        print(f"epoch {ep:3d}/{args.epochs}  loss {run_loss/seen:.4f}  "
              f"val_acc {va:.4f}  ({time.time()-t0:.0f}s){star}")

    torch.save({"model": args.model, "state_dict": model.state_dict(),
                "classes": CLASSES, "val_acc": history[-1]["val_acc"], "size": args.size,
                "epoch": args.epochs, "seed": args.seed},
               out_dir / f"weatherclf_{args.model}_last.pt")
    (out_dir / f"trainlog_{args.model}.json").write_text(
        json.dumps({"args": vars(args), "best_val_acc": best, "history": history},
                   indent=2), encoding="utf-8")
    print(f"\ndone. best val_acc {best:.4f} -> {out_dir / f'weatherclf_{args.model}_best.pt'}")


if __name__ == "__main__":
    main()
