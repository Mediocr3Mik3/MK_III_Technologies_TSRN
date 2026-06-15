"""Val-BPC A/B: EchoStateReservoir vs OscillatoryMemory.

Runs two short training jobs (same data, same seed) differing only in
--no-oscillatory vs --oscillatory, then reports validation BPC (bits per
character) = loss / ln(2) for each.

Usage (once shards exist):
    .venv312\\Scripts\\python.exe research\\_ab_reservoir_valbpc.py
"""
import subprocess, sys, os, json, time, tempfile, shutil

_RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

PYTHON = os.path.join(os.path.dirname(_RESEARCH_DIR), ".venv312", "Scripts", "python.exe")
TRAINER = os.path.join(_RESEARCH_DIR, "train_pretrain_local.py")
SHARD_DIR = r"E:\ml\tsrn_data\shards\pretrain"
STEPS = 500
CTX = 256
BATCH = 8

def run_variant(name: str, extra_flags: list):
    out_dir = tempfile.mkdtemp(prefix=f"ab_{name}_")
    log_file = os.path.join(out_dir, "log.txt")
    cmd = [
        PYTHON, TRAINER,
        "--data", SHARD_DIR,
        "--output", out_dir,
        "--steps", str(STEPS),
        "--context", str(CTX),
        "--batch", str(BATCH),
        "--val-every", str(STEPS),
        "--tag", f"ab_{name}",
    ] + extra_flags
    print(f"\n{'='*60}")
    print(f"Starting A/B arm: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        proc.wait()
    # grep last validation line
    val_loss = None
    with open(log_file) as lf:
        for line in lf:
            if "val loss" in line.lower() or "val_loss" in line.lower():
                # crude parser: look for a float after val loss
                parts = line.replace(",", " ").split()
                for p in parts:
                    try:
                        v = float(p)
                        if 0 < v < 20:
                            val_loss = v
                    except ValueError:
                        pass
    bpc = val_loss / 0.6931471805599453 if val_loss else None
    print(f"  {name:12s}: val_loss={val_loss}, BPC={bpc:.4f}" if bpc else f"  {name:12s}: no val loss found")
    return {"name": name, "val_loss": val_loss, "bpc": bpc, "out_dir": out_dir}


def main():
    if not os.path.isdir(SHARD_DIR):
        print(f"ERROR: Shard directory does not exist: {SHARD_DIR}")
        print("Build tokenized shards first (see E:/ml/tsrn_data/pipeline).")
        sys.exit(1)
    if not any(os.scandir(SHARD_DIR)):
        print(f"ERROR: Shard directory is empty: {SHARD_DIR}")
        print("Build tokenized shards first.")
        sys.exit(1)

    results = [
        run_variant("echo", []),
        run_variant("oscillatory", ["--oscillatory"]),
    ]

    print(f"\n{'='*60}")
    print("A/B SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['name']:12s}: val_loss={r['val_loss']:.4f}, BPC={r['bpc']:.4f}" if r['bpc'] else f"  {r['name']:12s}: N/A")
    if results[0]["bpc"] and results[1]["bpc"]:
        delta = results[1]["bpc"] - results[0]["bpc"]
        winner = "oscillatory" if delta < 0 else "echo"
        print(f"\n  Delta (osc - echo) BPC = {delta:+.4f}")
        print(f"  Winner: {winner}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
