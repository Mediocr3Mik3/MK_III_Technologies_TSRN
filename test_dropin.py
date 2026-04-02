"""
Smoke test for the drop-in TropFormer implementation.
Verifies API compatibility with torch.nn.Transformer* modules.
"""

import torch
import torch.nn as nn
from tropformer_dropin import (
    TropicalEncoderLayer, TropicalDecoderLayer,
    TropicalEncoder, TropicalDecoder, TropicalTransformer,
    TropicalEncoderModel, TropicalDecoderModel,
    TropicalViT, TropicalViTForClassification,
    TropicalKVCache, make_causal_mask,
    maslov_summary, lf_blend_summary,
)

torch.manual_seed(42)
B, L, D = 2, 8, 64
H = 4
FFN = 128

print("=" * 60)
print("  TropFormer Drop-in Smoke Tests")
print("=" * 60)

# Test 1: TropicalEncoderLayer matches nn.TransformerEncoderLayer API
print("\n[1] TropicalEncoderLayer (drop-in for nn.TransformerEncoderLayer)")
enc_layer = TropicalEncoderLayer(D, H, FFN, dropout=0.0)
src = torch.randn(B, L, D)
out = enc_layer(src)
assert out.shape == (B, L, D), f"Expected {(B, L, D)}, got {out.shape}"

# With masks
padding_mask = torch.zeros(B, L, dtype=torch.bool)
padding_mask[0, -2:] = True  # mask last 2 tokens of first sample
out_masked = enc_layer(src, src_key_padding_mask=padding_mask)
assert out_masked.shape == (B, L, D)
print(f"  OK: output shape {out.shape}, masked shape {out_masked.shape}")

# Test 2: TropicalDecoderLayer
print("\n[2] TropicalDecoderLayer (drop-in for nn.TransformerDecoderLayer)")
dec_layer = TropicalDecoderLayer(D, H, FFN, dropout=0.0)
tgt = torch.randn(B, 4, D)
memory = torch.randn(B, L, D)
causal = make_causal_mask(4, tgt.device)
out = dec_layer(tgt, memory, tgt_mask=causal)
assert out.shape == (B, 4, D), f"Expected {(B, 4, D)}, got {out.shape}"
print(f"  OK: output shape {out.shape}")

# Test 3: TropicalEncoder stack
print("\n[3] TropicalEncoder (stack of N layers)")
enc = TropicalEncoder(
    TropicalEncoderLayer(D, H, FFN, dropout=0.0),
    num_layers=3,
    norm=nn.LayerNorm(D),
)
out = enc(src)
assert out.shape == (B, L, D)
print(f"  OK: 3-layer encoder output {out.shape}")

# Test 4: TropicalTransformer (encoder-decoder)
print("\n[4] TropicalTransformer (encoder + decoder)")
transformer = TropicalTransformer(
    d_model=D, nhead=H, num_encoder_layers=2, num_decoder_layers=2,
    dim_feedforward=FFN, dropout=0.0,
)
out = transformer(src, tgt)
assert out.shape == (B, 4, D)
print(f"  OK: transformer output {out.shape}")

# Test 5: TropicalEncoderModel (BERT-style)
print("\n[5] TropicalEncoderModel (BERT-style)")
bert = TropicalEncoderModel(vocab_size=100, d_model=D, nhead=H, num_layers=2,
                            dim_feedforward=FFN, dropout=0.0, max_seq_len=32)
ids = torch.randint(0, 100, (B, 12))
out = bert(ids)
assert out.shape == (B, 12, D)
print(f"  OK: BERT output {out.shape}")

# Test 6: TropicalDecoderModel (GPT-style)
print("\n[6] TropicalDecoderModel (GPT-style)")
gpt = TropicalDecoderModel(vocab_size=100, d_model=D, nhead=H, num_layers=2,
                           dim_feedforward=FFN, dropout=0.0, max_seq_len=32)
out = gpt(ids)
assert out.shape == (B, 12, 100), f"Expected {(B, 12, 100)}, got {out.shape}"
print(f"  OK: GPT output {out.shape} (logits over vocab)")

# Test 7: TropicalViT
print("\n[7] TropicalViT (vision backbone)")
vit = TropicalViT(img_size=28, patch_size=7, in_channels=1,
                   d_model=D, nhead=H, num_layers=2,
                   dim_feedforward=FFN, dropout=0.0)
img = torch.randn(B, 1, 28, 28)
out = vit(img)
assert out.shape == (B, 17, D)  # 16 patches + 1 CLS
print(f"  OK: ViT output {out.shape} (17 = 16 patches + CLS)")

# Test 8: TropicalViTForClassification
print("\n[8] TropicalViTForClassification")
clf = TropicalViTForClassification(
    num_classes=10, img_size=28, patch_size=7, in_channels=1,
    d_model=D, nhead=H, num_layers=2, dim_feedforward=FFN, dropout=0.0,
)
logits = clf(img)
assert logits.shape == (B, 10)
print(f"  OK: classifier output {logits.shape}, params={clf.count_params():,}")

# Test 9: KV Cache
print("\n[9] TropicalKVCache")
cache = TropicalKVCache()
k1 = torch.randn(B, H, 1, D // H)
v1 = torch.randn(B, H, 1, D // H)
k_all, v_all = cache.update(k1, v1)
assert k_all.shape == (B, H, 1, D // H)
k2 = torch.randn(B, H, 1, D // H)
v2 = torch.randn(B, H, 1, D // H)
k_all, v_all = cache.update(k2, v2)
assert k_all.shape == (B, H, 2, D // H)
print(f"  OK: cache grows {(B,H,1,D//H)} -> {k_all.shape}")

# Test 10: Diagnostics
print("\n[10] Diagnostic utilities")
temps = maslov_summary(clf)
blends = lf_blend_summary(clf)
print(f"  OK: {len(temps)} temperature modules, {len(blends)} blend gates")

# Test 11: Causal mask
print("\n[11] make_causal_mask")
mask = make_causal_mask(4, torch.device("cpu"))
assert mask.shape == (4, 4)
assert mask[0, 0] == 0.0 and mask[0, 1] == -1e9
print(f"  OK: shape {mask.shape}, [0,0]={mask[0,0]:.0f}, [0,1]={mask[0,1]:.0f}")

# Test 12: batch_first=False compatibility
print("\n[12] batch_first=False (seq-first convention)")
enc_sf = TropicalEncoderLayer(D, H, FFN, dropout=0.0, batch_first=False)
src_sf = torch.randn(L, B, D)  # (L, B, D) seq-first
out_sf = enc_sf(src_sf)
assert out_sf.shape == (L, B, D)
print(f"  OK: seq-first input {src_sf.shape} -> output {out_sf.shape}")

# Test 13: Gradient flow
print("\n[13] Gradient flow check")
clf.train()
logits = clf(img)
loss = logits.sum()
loss.backward()
n_grads = sum(1 for p in clf.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
n_params = sum(1 for p in clf.parameters() if p.requires_grad)
print(f"  OK: {n_grads}/{n_params} parameters received gradients")

# Test 14: Serialization
print("\n[14] torch.save / torch.load")
import os
save_path = "_test_dropin_tmp.pt"
torch.save(clf.state_dict(), save_path)
sz = os.path.getsize(save_path)
clf2 = TropicalViTForClassification(
    num_classes=10, img_size=28, patch_size=7, in_channels=1,
    d_model=D, nhead=H, num_layers=2, dim_feedforward=FFN, dropout=0.0,
)
clf2.load_state_dict(torch.load(save_path, weights_only=True))
os.remove(save_path)
print(f"  OK: saved ({sz:,} bytes) and loaded successfully")

print("\n" + "=" * 60)
print("  ALL 14 TESTS PASSED")
print("=" * 60)
