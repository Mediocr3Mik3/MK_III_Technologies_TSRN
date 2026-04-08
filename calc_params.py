import sys, os
sys.path.insert(0, '.')
from tsrn_dml import TSRN, load_enwik8

# Test different configs to reach ~50M params
dataset = load_enwik8(context_len=256)

configs = [
    {'d_model': 512, 'n_blocks': 1, 'n_heads': 8},
    {'d_model': 512, 'n_blocks': 2, 'n_heads': 8},
    {'d_model': 512, 'n_blocks': 3, 'n_heads': 8},  # baseline: 22.6M
    {'d_model': 512, 'n_blocks': 4, 'n_heads': 8},
    {'d_model': 512, 'n_blocks': 5, 'n_heads': 8},
    {'d_model': 512, 'n_blocks': 6, 'n_heads': 8},
]

for cfg in configs:
    model = TSRN(vocab=dataset.vocab_sz, d_model=cfg['d_model'], 
                 context_len=256, n_blocks=cfg['n_blocks'], 
                 n_heads=cfg['n_heads'], top_k=16, mem_depth=7, dropout=0.0)
    params = model.count_params()
    print(f"d_model={cfg['d_model']}, n_blocks={cfg['n_blocks']}, n_heads={cfg['n_heads']}: {params:,} params")
