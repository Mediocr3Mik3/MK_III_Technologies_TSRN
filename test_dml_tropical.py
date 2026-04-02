"""Test DirectML compatibility with TropFormer/DTN training."""
import torch
import torch_directml
import time

dml = torch_directml.device()

# Test 1: TropFormer forward on DML
print("Testing TropFormer forward on DML...")
from tropformer import TropFormer
m = TropFormer(
    img_size=32, patch_size=8, in_channels=3, num_classes=10,
    d_model=128, num_heads=4, num_layers=4, ffn_dim=256,
).to(dml)
x = torch.randn(32, 3, 32, 32).to(dml)
out = m(x)
print(f"  Forward OK: {out.shape}")

print("  Trying backward...")
try:
    out.sum().backward()
    print("  Backward OK!")
except Exception as e:
    print(f"  Backward FAILED: {type(e).__name__}: {str(e)[:200]}")

# Test 2: DTN forward on DML
print("\nTesting DTN forward on DML...")
from deep_tropical_net import DeepTropNet
m2 = DeepTropNet(
    img_size=32, patch_size=8, in_channels=3, num_classes=10,
    d_model=128, num_layers=4, num_heads=4, ffn_dim=256,
).to(dml)
x2 = torch.randn(32, 3, 32, 32).to(dml)
out2 = m2(x2)
print(f"  Forward OK: {out2.shape}")

print("  Trying backward...")
try:
    out2.sum().backward()
    print("  Backward OK!")
except Exception as e:
    print(f"  Backward FAILED: {type(e).__name__}: {str(e)[:200]}")

# Test 3: Classical transformer (should fully work)
print("\nTesting Classical Transformer on DML...")
from classical_transformer import ClassicalTransformer
m3 = ClassicalTransformer(
    img_size=32, patch_size=8, in_channels=3, num_classes=10,
    d_model=128, num_heads=4, num_layers=4, ffn_dim=256, dropout=0.1,
).to(dml)
x3 = torch.randn(32, 3, 32, 32).to(dml)
out3 = m3(x3)
print(f"  Forward OK: {out3.shape}")

print("  Trying backward...")
try:
    out3.sum().backward()
    print("  Backward OK!")
except Exception as e:
    print(f"  Backward FAILED: {type(e).__name__}: {str(e)[:200]}")

# Speed comparison at larger scale
print("\n=== Speed comparison (d=256, L=6, batch=128) ===")
m_big = ClassicalTransformer(
    img_size=32, patch_size=8, in_channels=3, num_classes=10,
    d_model=256, num_heads=8, num_layers=6, ffn_dim=512, dropout=0.1,
)
xb = torch.randn(128, 3, 32, 32)
yb = torch.randint(0, 10, (128,))

# CPU
opt_cpu = torch.optim.AdamW(m_big.parameters(), lr=1e-3)
# warmup
o = m_big(xb); o.sum().backward(); m_big.zero_grad()
times = []
for _ in range(5):
    t0 = time.time()
    opt_cpu.zero_grad()
    o = m_big(xb)
    torch.nn.functional.cross_entropy(o, yb).backward()
    opt_cpu.step()
    times.append(time.time() - t0)
print(f"  CPU avg step: {sum(times)/5:.3f}s")

# DML
m_big_dml = ClassicalTransformer(
    img_size=32, patch_size=8, in_channels=3, num_classes=10,
    d_model=256, num_heads=8, num_layers=6, ffn_dim=512, dropout=0.1,
).to(dml)
xb_dml = xb.to(dml)
yb_dml = yb.to(dml)
opt_dml = torch.optim.AdamW(m_big_dml.parameters(), lr=1e-3)
# warmup
o = m_big_dml(xb_dml); o.sum().backward(); m_big_dml.zero_grad()
times = []
for _ in range(5):
    t0 = time.time()
    opt_dml.zero_grad()
    o = m_big_dml(xb_dml)
    torch.nn.functional.cross_entropy(o, yb_dml).backward()
    opt_dml.step()
    times.append(time.time() - t0)
print(f"  DML avg step: {sum(times)/5:.3f}s")
