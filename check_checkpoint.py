# save as check_checkpoint.py in your TropFormer directory
import torch
import torch.nn as nn
import math, sys, os
sys.path.insert(0, ".")

from tsrn_dml import TSRN, load_enwik8

ckpt_path = "checkpoints/tsrn_enwik8_v6_100k_80000steps.pt"  # adjust to latest

print(f"Loading: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = ckpt["config"]

print(f"\nStored best_val_bpc: {ckpt.get('best_val_bpc', 'NOT FOUND')}")
print(f"Stored step: {ckpt.get('step', 'NOT FOUND')}")

# Build model on CPU
model = TSRN(
    vocab=cfg["vocab"], d_model=cfg["d_model"],
    context_len=cfg["context_len"], n_blocks=cfg["n_blocks"],
    top_k=cfg["top_k"], n_heads=cfg["n_heads"],
    mem_depth=cfg["mem_depth"], dropout=0.0
)

# Handle broken weight tying from DirectML training
sd = dict(ckpt["model_state_dict"])
head_w = sd.pop("head.weight", None)
if head_w is not None and "embed.weight" in sd:
    if not torch.allclose(head_w, sd["embed.weight"]):
        print(f"\n  NOTE: Weight tying was broken during DirectML training")
        print(f"    embed norm={sd['embed.weight'].norm():.4f}  head norm={head_w.norm():.4f}")

missing, unexpected = model.load_state_dict(sd, strict=False)
real_missing = [k for k in (missing or []) if k != "head.weight"]
print(f"Missing keys:    {real_missing if real_missing else 'none'}")
print(f"Unexpected keys: {unexpected if unexpected else 'none'}")

# Restore head.weight as separate parameter (untied)
if head_w is not None:
    model.head.weight = nn.Parameter(head_w.clone())

# Check 1: _cached_v buffer
print(f"\n=== Reservoir buffer check ===")
for name, module in model.named_modules():
    if hasattr(module, '_cached_v'):
        is_registered = '_cached_v' in dict(module.named_buffers())
        is_plain = '_cached_v' in module.__dict__
        norm = module._cached_v.norm().item()
        all_zero = torch.all(module._cached_v == 0).item()
        print(f"  {name}._cached_v:")
        print(f"    registered buffer: {is_registered}")
        print(f"    plain attribute:   {is_plain}")
        print(f"    norm:              {norm:.4f}")
        print(f"    all zeros:         {all_zero}")
        if all_zero:
            print(f"    !! PROBLEM: buffer is zeros — fix not working")
        elif not is_registered:
            print(f"    !! PROBLEM: not a registered buffer — won't save")
        else:
            print(f"    OK: non-zero registered buffer")

# Check 2: are weights trained (not random)?
print(f"\n=== Weight training check ===")
embed_norm = model.embed.weight.norm().item()
head_norm = model.head.weight.norm().item()
print(f"  embed.weight norm:  {embed_norm:.4f}  (random init ~{model.d**0.5:.1f})")
print(f"  head.weight norm:   {head_norm:.4f}")
print(f"  Weight tied: {model.head.weight.data_ptr() == model.embed.weight.data_ptr()}")

for i, block in enumerate(model.s1_blocks):
    attn_norm = block.attn.Wq.weight.norm().item()
    sheaf_alpha = block.sheaf.alpha.item()
    print(f"  s1_block[{i}] Wq norm: {attn_norm:.4f}, sheaf.alpha: {sheaf_alpha:.6f}")

for i, block in enumerate(model.s2_blocks):
    attn_norm = block.attn.Wq.weight.norm().item()
    print(f"  s2_block[{i}] Wq norm: {attn_norm:.4f}")

# Check 3: reservoir learned parameters
print(f"\n=== Reservoir learned params ===")
for block in model.s1_blocks:
    if block.use_reservoir:
        res = block.reservoir
        print(f"  log_rho: {res.log_rho.item():.6f}")
        print(f"  rho_target: {(torch.sigmoid(res.log_rho)*1.5).item():.6f}")
        print(f"  leak: {torch.sigmoid(res.leak).item():.6f}")
        print(f"  readout norm: {res.readout.weight.norm().item():.6f}")
        print(f"  W_res norm: {res.W_res.norm().item():.6f}")
        print(f"  W_res sparsity: {(res.W_res==0).float().mean().item():.3f}")

# Check 4: forward pass on CPU (tiny batch, slow but correct)
print(f"\n=== CPU forward pass sanity check ===")
print(f"  Loading dataset...")
dataset = load_enwik8(context_len=cfg["context_len"])
device = torch.device("cpu")

torch.manual_seed(42)
x, y = dataset.batch("test", 50, device)  # tiny batch, CPU only

model.eval()
with torch.no_grad():
    _, loss = model(x, y)

bpc = loss.item() / math.log(2)
print(f"  Test BPC (CPU, 50 samples): {bpc:.4f}")
if bpc < 1.5:
    print(f"  ✅ PASS — weights loaded correctly")
elif bpc < 3.0:
    print(f"  ⚠️  MARGINAL — weights partially loaded, something off")
else:
    print(f"  ❌ FAIL — BPC {bpc:.4f} indicates wrong weights or broken arch")

print(f"\n=== Summary ===")
total_params = sum(p.numel() for p in model.parameters())
total_buffers = sum(b.numel() for b in model.buffers())
print(f"  Total parameters: {total_params:,}")
print(f"  Total buffers:    {total_buffers:,}")
print(f"  State dict keys:  {len(ckpt['model_state_dict'])}")
print(f"  Model state keys: {len(model.state_dict())}")
keys_match = set(ckpt['model_state_dict'].keys()) == set(model.state_dict().keys())
print(f"  Keys match exactly: {keys_match}")