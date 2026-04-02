"""Speed comparison: CPU vs DirectML for all three architectures."""
import torch
import torch.nn.functional as F
import time
import torch_directml

from tropformer import TropFormer, _USE_SMOOTH_MAX
import tropformer
from deep_tropical_net import DeepTropNet
from classical_transformer import ClassicalTransformer

dml = torch_directml.device()
tropformer._USE_SMOOTH_MAX = True  # Enable DML compat

D, H, L, FFN = 128, 4, 4, 256
BS = 64
STEPS = 10

configs = [
    ("Classical", lambda dev: ClassicalTransformer(
        img_size=32, patch_size=8, in_channels=3, num_classes=10,
        d_model=D, num_heads=H, num_layers=L, ffn_dim=FFN, dropout=0.1).to(dev)),
    ("TropFormer", lambda dev: TropFormer(
        img_size=32, patch_size=8, in_channels=3, num_classes=10,
        d_model=D, num_heads=H, num_layers=L, ffn_dim=FFN, dropout=0.1).to(dev)),
    ("DTN", lambda dev: DeepTropNet(
        img_size=32, patch_size=8, in_channels=3, num_classes=10,
        d_model=D, num_layers=L, num_heads=H, ffn_dim=FFN, dropout=0.1).to(dev)),
]

print(f"{'Model':>15}  {'CPU':>8}  {'DML':>8}  {'Speedup':>8}")
print("-" * 50)

for name, make_model in configs:
    for dev_name, dev in [("cpu", torch.device("cpu")), ("dml", dml)]:
        torch.manual_seed(42)
        model = make_model(dev)
        x = torch.randn(BS, 3, 32, 32).to(dev)
        y = torch.randint(0, 10, (BS,)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Warmup
        o = model(x)
        F.cross_entropy(o, y).backward()
        opt.step()
        opt.zero_grad()
        
        # Time
        times = []
        for _ in range(STEPS):
            t0 = time.time()
            opt.zero_grad()
            o = model(x)
            loss = F.cross_entropy(o, y)
            loss.backward()
            opt.step()
            times.append(time.time() - t0)
        
        avg = sum(times) / len(times)
        if dev_name == "cpu":
            cpu_time = avg
        else:
            dml_time = avg
    
    speedup = cpu_time / dml_time
    print(f"{name:>15}  {cpu_time:>7.3f}s  {dml_time:>7.3f}s  {speedup:>7.2f}x")
