"""Quick script to evaluate MWPM baseline on toric code syndrome decoding."""
from syndrome_data import mwpm_evaluate
import time

error_rates = [0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]

for d in [3, 5, 7]:
    print(f"\n=== d={d} ===")
    for p in error_rates:
        t0 = time.time()
        r = mwpm_evaluate(d, p, n_samples=10000)
        dt = time.time() - t0
        print(f"  p={p:.2f}: acc={r['accuracy']:.4f}  "
              f"p_L={r['logical_error_rate']:.4f}  "
              f"raw={r['raw_logical_rate']:.4f}  ({dt:.1f}s)")
