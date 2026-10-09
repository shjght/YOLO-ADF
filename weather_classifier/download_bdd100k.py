# -*- coding: utf-8 -*-
"""BDD100K downloader - only for a FRESH machine.

On both the local mirror (F:/xlw) and the training machine (C:/Users/KUST/xl)
the dataset already exists, so normally this script just detects it and exits.

What this benchmark actually needs:
  1. label JSONs  (bdd100k_labels_images_{train,val}.json, ~300 MB)  - weather attributes
  2. image jpgs   (bdd100k images 100k train/val, ~5.3 GB)

Sources tried in order:
  A. existing local copy (auto-detect)
  B. ETH Zurich public mirror  https://dl.cv.ethz.ch/bdd100k/data/   (no login)
  C. official site (registration required, manual):
       https://bdd-data.berkeley.edu  /  https://scal.ai  -> "BDD100K" -> Download
  D. Kaggle mirror via kagglehub:  kagglehub.dataset_download("solesensei/solesensei_bdd100k")
     (needs ~/.kaggle/kaggle.json API token)

    python download_bdd100k.py                  # detect only
    python download_bdd100k.py --labels         # fetch label JSONs only (enough if images exist)
    python download_bdd100k.py --with-images    # also fetch the 5.3 GB image archive
"""

import argparse
import zipfile
from pathlib import Path

import urllib.request

HERE = Path(__file__).resolve().parent
ETH_BASE = "https://dl.cv.ethz.ch/bdd100k/data/"
LABEL_ZIP = "bdd100k_labels_release.zip"
IMAGE_ZIP = "bdd100k_images_100k.zip"

CANDIDATE_ROOTS = [HERE.parent, Path("F:/xlw"), Path("C:/Users/KUST/xl"),
                   Path("C:/Users/KUST/xlw")]


def detect_existing():
    for root in CANDIDATE_ROOTS:
        labels = root / "bdd100k_final" / "labels"
        need = ["bdd100k_labels_images_train.json", "bdd100k_labels_images_val.json"]
        if all((labels / n).exists() for n in need):
            n_train = len(list((root / "bdd100k_final" / "images" / "100k" / "train").glob("*.jpg"))) \
                if (root / "bdd100k_final" / "images" / "100k" / "train").exists() else -1
            print(f"[ok] found existing copy at {root}")
            print(f"     labels: train+val json present; train jpgs on disk: {n_train}")
            return True
    return False


def download(url: str, out: Path):
    """Stream with resume support."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    done = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except Exception as e:
        if done and "206" not in str(resp.status if hasattr(resp, 'status') else ''):
            print(f"[warn] {e}; restarting from scratch")
            done, tmp = 0, out.with_suffix(out.suffix + ".part")
        else:
            raise
    total = resp.headers.get("Content-Length")
    total = int(total) + done if total else None
    mode = "ab" if done else "wb"
    with tmp.open(mode) as f, resp:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e6:.0f}/{total/1e6:.0f} MB "
                      f"({100*done/total:.1f}%)", end="", flush=True)
    print()
    tmp.rename(out)
    return out


def try_eth(fname: str, dest_dir: Path):
    url = ETH_BASE + fname
    print(f"[eth] trying {url}")
    try:
        out = download(url, dest_dir / fname)
        print(f"[eth] extracting {out.name} ...")
        with zipfile.ZipFile(out) as z:
            z.extractall(dest_dir)
        return True
    except Exception as e:
        print(f"[eth] failed: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="store_true", help="fetch label jsons only")
    ap.add_argument("--with-images", action="store_true", help="also fetch 5.3 GB image zip")
    ap.add_argument("--dest", default=str(HERE.parent / "bdd100k_final_download"))
    args = ap.parse_args()

    if detect_existing():
        print("\nnothing to download. run build_labels.py next.")
        return

    dest = Path(args.dest)
    print(f"\n[1/2] label JSONs -> {dest}")
    ok = try_eth(LABEL_ZIP, dest)
    if not ok:
        print("ETH mirror unreachable. Pick one:")
        print("  - official (free, registration): https://bdd-data.berkeley.edu "
              "or https://scal.ai -> download bdd100k_labels_release.zip")
        print("  - kaggle mirror: pip install kagglehub && "
              'python -c "import kagglehub;'
              ' print(kagglehub.dataset_download(\'solesensei/solesensei_bdd100k\'))"')
        return
    # the release zip unpacks to bdd100k/labels/... - copy the two jsons up
    for j in (dest).rglob("bdd100k_labels_images_*.json"):
        tgt = dest / "labels"
        tgt.mkdir(exist_ok=True)
        j.rename(tgt / j.name)
        print(f"  placed {tgt / j.name}")

    if args.with_images:
        print(f"\n[2/2] images 100k -> {dest}")
        if not try_eth(IMAGE_ZIP, dest):
            print("manual fallback: official site / kaggle (see above)")

    print("\ndone. then: python build_labels.py --data-root", dest.parent)


if __name__ == "__main__":
    main()
