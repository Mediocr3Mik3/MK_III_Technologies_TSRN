import torch
import math

# Old run (echo reservoir)
ckpt_old = torch.load(r'D:\ml\tsrn_data\checkpoints\20260614_v2_kleene_directml_d256_ctx256_best.pt', map_location='cpu', weights_only=False)
log_old = ckpt_old.get('log', [])
print('=== OLD RUN (Echo reservoir) ===')
for e in log_old:
    print(f"  step {e['step']:>5}: val_loss={e['val_loss']:.4f}  val_bpc={e['val_bpc']:.4f}")

print()

# Oscillatory run
path = r'D:\ml\tsrn_data\checkpoints\20260615_v2_kleene_directml_osc_d256_ctx256_best.pt'
try:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    log = ckpt.get('log', [])
    print(f'=== OSC RUN ===')
    print(f"  step {ckpt.get('step', '?')}: best_val_bpc={ckpt.get('best_val_bpc', '?')}")
    for e in log:
        print(f"  step {e['step']:>5}: val_loss={e['val_loss']:.4f}  val_bpc={e['val_bpc']:.4f}")
except Exception as ex:
    print(f'  ERROR loading {path}: {ex}')

# Projection math
print()
print('=== PROJECTION ===')
steps = [e['step'] for e in log_old]
losses = [e['val_loss'] for e in log_old]

import numpy as np
x = np.array(steps, dtype=float)
y = np.array(losses, dtype=float)

# Power law fit: loss = a * x^(-b) + c
# Try different floor values and pick best fit
best_r2 = -1
best_fit = None
for c in np.arange(1.2, 2.2, 0.05):
    y_shifted = y - c
    mask = y_shifted > 0
    if mask.sum() < 3:
        continue
    log_x = np.log(x[mask])
    log_y_shifted = np.log(y_shifted[mask])
    # linear fit: log(y-c) = log(a) - b*log(x)
    coeffs = np.polyfit(log_x, log_y_shifted, 1)
    b_fit = -coeffs[0]
    a_fit = math.exp(coeffs[1])
    y_pred = a_fit * x**(-b_fit) + c
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    if r2 > best_r2:
        best_r2 = r2
        best_fit = (a_fit, b_fit, c, r2)

if best_fit:
    a_fit, b_fit, c_fit, r2 = best_fit
    print(f"Best fit: loss = {a_fit:.3f} * step^(-{b_fit:.4f}) + {c_fit:.3f}")
    print(f"R^2 = {r2:.4f}")
    print()
    
    # Thresholds for 18M param model on mixed data
    # Based on Chinchilla scaling laws and empirical small-model behavior:
    thresholds = [
        (2.0, 'basic phrases', 'short coherent phrases, some grammar'),
        (1.5, 'sentences', 'grammatically correct sentences'),
        (1.2, 'paragraphs', 'topic-consistent multi-sentence text'),
        (1.0, 'fluent', 'natural, coherent paragraphs'),
    ]
    for target_loss, label, desc in thresholds:
        if target_loss <= c_fit:
            print(f"  Target loss={target_loss} ({label}): UNREACHABLE — below estimated floor {c_fit:.2f}")
        else:
            t = (a_fit / (target_loss - c_fit)) ** (1 / b_fit)
            print(f"  Target loss={target_loss} ({label}): ~{t:,.0f} steps")
            print(f"    ({desc})")
    
    # Current trajectory
    print()
    print('=== CURRENT TRAJECTORY ===')
    for s in [10000, 25000, 50000, 100000]:
        pred = a_fit * s**(-b_fit) + c_fit
        print(f"  step {s:>6}: projected val_loss = {pred:.3f}  (val_bpc = {pred/math.log(2):.3f})")
else:
    print('Could not find valid power-law fit.')
