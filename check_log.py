import json
with open('results/tsrn_enwik8_v2_100k_100000steps.json', 'r') as f:
    data = json.load(f)

log = data['training_log']
print('Last 10 training entries:')
for entry in log[-10:]:
    print(f"Step {entry['step']:>6}: val_bpc={entry['val_bpc']:.4f}, train_bpc={entry['train_bpc']:.4f}")
