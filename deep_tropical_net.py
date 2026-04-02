"""
Deep Tropical Network (v2 -- Hybrid)
=====================================
Hybrid tropical-classical architecture built on TropFormer components.

Key difference from TropFormer:
  - Gates biased TOWARD tropical (sigmoid(+2) ~ 0.88 tropical weight)
    vs TropFormer which biases toward classical (sigmoid(-2) ~ 0.12)
  - TropicalLinear used in more positions (Q/K projections, FFN)
  - TropicalBatchNorm for polytope stability between blocks
  - Deeper stacking with tropical residual option

The DTN uses TropFormer's proven hybrid attention (gated tropical + classical
scores) and hybrid FFN (gated tropical + classical branches) but configured
to prefer tropical computation. The gate can still learn to use classical
paths where needed for training stability.

Based on Roadmap.md Section 3 and TropFormer_Benchmarks.md.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tropformer import (
    TropicalLinear,
    TropicalDropout,
    MaslovTemperature,
    LFDualActivation,
    _tropical_max,
)


# =============================================================================
# DTN-Specific Primitives
# =============================================================================

class TropicalBatchNorm(nn.Module):
    """
    Tropical batch normalization to prevent polytope collapse/explosion.
    
    Classical BN: (x - mean) / std
    Tropical BN:  (x - trop_max) / trop_range
    
    trop_max   = max over batch of max(x)   -- the "tropical mean"
    trop_range = max(x) - min(x)            -- spread of polytope cells
    
    After normalization, the most active feature = 0 (tropical one),
    and spread is standardized to ~1.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer('running_max', torch.zeros(num_features))
        self.register_buffer('running_range', torch.ones(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x_flat = x.reshape(-1, self.num_features)
            with torch.no_grad():
                trop_max = x_flat.max(dim=0).values
                trop_min = x_flat.min(dim=0).values
                trop_range = (trop_max - trop_min).clamp(min=self.eps)
                self.running_max = (1 - self.momentum) * self.running_max + self.momentum * trop_max
                self.running_range = (1 - self.momentum) * self.running_range + self.momentum * trop_range
        else:
            trop_max = self.running_max
            trop_range = self.running_range
        
        x_norm = (x - trop_max) / (trop_range + self.eps)
        return self.gamma * x_norm + self.beta


# =============================================================================
# DTN Hybrid Attention (TropFormer-based, tropical-biased)
# =============================================================================

class DTNAttention(nn.Module):
    """
    Hybrid tropical-classical attention, biased toward tropical.
    
    Uses TropFormer's proven gated score mechanism:
      scores = g * trop_scores + (1-g) * class_scores
    But with gate initialized toward tropical (g ~ 0.88 at init).
    
    Also uses TropicalLinear for Q/K projections (not just classical),
    making the projections themselves tropical while keeping the gated
    score fallback for training stability.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        init_temp: float = 1.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self._scale = self.d_k ** -0.5
        
        # Q/K: TropicalLinear for tropical-first projections
        self.q_proj = TropicalLinear(d_model, d_model)
        self.k_proj = TropicalLinear(d_model, d_model)
        # V: classical for smooth gradient flow in values
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Score gate: biased TOWARD tropical (opposite of TropFormer)
        self.score_gate = nn.Linear(d_model, num_heads)
        nn.init.zeros_(self.score_gate.weight)
        nn.init.constant_(self.score_gate.bias, +2.0)  # sigmoid(+2)~0.88 tropical
        
        # Maslov temperature
        self.maslov = MaslovTemperature(num_heads, init_temp)
        self.attn_dropout = nn.Dropout(dropout)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, x: torch.Tensor, mask=None) -> tuple:
        B, L, D = x.shape
        
        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))
        
        # Classical scores
        class_scores = torch.matmul(Q, K.transpose(-2, -1)) * self._scale
        
        # Tropical scores: max-plus inner product
        trop_scores = _tropical_max(
            Q.unsqueeze(3) + K.unsqueeze(2), dim=-1
        ) * self._scale
        
        # Gate (biased toward tropical)
        g = torch.sigmoid(self.score_gate(x))     # (B, L, H)
        g = g.permute(0, 2, 1).unsqueeze(-1)      # (B, H, L, 1)
        scores = g * trop_scores + (1.0 - g) * class_scores
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = self.maslov(scores)
        attn = self.attn_dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out), attn


# =============================================================================
# DTN Hybrid FFN (TropFormer-based, tropical-biased)
# =============================================================================

class DTNHybridFFN(nn.Module):
    """
    Hybrid FFN with parallel tropical and classical branches,
    gate biased toward the tropical path.
    
    Tropical: TropicalLinear -> LFDualActivation -> Dropout
    Classical: Linear -> GELU
    Fusion: g * trop + (1-g) * classical  (g init ~0.88 tropical)
    """
    
    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        dropout: float = 0.1,
        lf_pieces: int = 8,
        lf_mode: str = "blend",
    ):
        super().__init__()
        self.trop_up = TropicalLinear(d_model, ffn_dim)
        self.lf_act = LFDualActivation(ffn_dim, num_pieces=lf_pieces, mode=lf_mode)
        self.trop_drop = nn.Dropout(dropout)
        
        self.class_up = nn.Linear(d_model, ffn_dim)
        self.gelu = nn.GELU()
        
        # Gate biased TOWARD tropical
        self.gate_proj = nn.Linear(d_model, ffn_dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, +2.0)  # sigmoid(+2)~0.88 tropical
        
        self.down_proj = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trop = self.trop_drop(self.lf_act(self.trop_up(x)))
        classical = self.gelu(self.class_up(x))
        g = torch.sigmoid(self.gate_proj(x))
        fused = g * trop + (1.0 - g) * classical
        return self.dropout(self.down_proj(fused))


# =============================================================================
# DTN Block (attention + FFN, pre-norm)
# =============================================================================

class DTNBlock(nn.Module):
    """
    Single DTN transformer block:
      x -> PreNorm -> DTNAttention -> + x   (attention sub-layer)
        -> PreNorm -> DTNHybridFFN -> + x   (FFN sub-layer)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        lf_pieces: int = 8,
        lf_mode: str = "blend",
        init_temp: float = 1.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = DTNAttention(d_model, num_heads, dropout, init_temp)
        self.drop1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = DTNHybridFFN(d_model, ffn_dim, dropout, lf_pieces, lf_mode)
    
    def forward(self, x: torch.Tensor, mask=None) -> tuple:
        attn_out, attn_w = self.attn(self.norm1(x), mask)
        x = x + self.drop1(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x, attn_w


# =============================================================================
# Deep Tropical Network (Hybrid, v2)
# =============================================================================

class DeepTropNet(nn.Module):
    """
    Hybrid Deep Tropical Network built on TropFormer components.
    
    Architecture:
        Image -> patchify -> embed -> [CLS + pos]
              -> TropicalBatchNorm (enter tropical domain)
              -> [DTNBlock x num_layers]
              -> LayerNorm -> head(CLS) -> logits
    
    Key differences from TropFormer:
      1. Gates biased toward tropical (sigmoid(+2) ~ 0.88) not classical
      2. TropicalLinear for Q/K projections (tropical routing in projections)
      3. TropicalBatchNorm at input (polytope stabilization)
      4. Gate can still learn to use classical where needed (hybrid safety)
    
    Supports both vision (patch embedding) and sequence inputs.
    """
    
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        num_classes: int = 10,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = None,
        lf_pieces: int = 8,
        lf_mode: str = "blend",
        init_temp: float = 1.0,
        dropout: float = 0.1,
        use_trop_bn: bool = True,
        # Legacy compat params (ignored)
        num_attn_layers: int = 0,
        trop_dropout: float = 0.05,
        ste_temp: float = 1.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        ffn_dim = ffn_dim or d_model * 2
        
        # Input: classical embedding
        self.patch_embed = nn.Linear(patch_dim, d_model)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        
        # CLS token and positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.embed_drop = nn.Dropout(dropout)
        
        # TropicalBatchNorm at input (enter tropical domain)
        self.use_trop_bn = use_trop_bn
        if use_trop_bn:
            self.trop_bn = TropicalBatchNorm(d_model)
        
        # DTN blocks (hybrid attention + FFN, tropical-biased)
        self.blocks = nn.ModuleList([
            DTNBlock(d_model, num_heads, ffn_dim, dropout,
                     lf_pieces, lf_mode, init_temp)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.head = nn.Linear(d_model, num_classes)
        nn.init.zeros_(self.head.bias)
    
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.view(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(B, -1, C * p * p)
    
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def maslov_summary(self) -> dict:
        """Per-block, per-head Maslov temperatures after training."""
        return {
            f"block_{i}": block.attn.maslov.temperatures.detach().cpu()
            for i, block in enumerate(self.blocks)
        }
    
    def gate_summary(self) -> dict:
        """Mean score gate values (higher = more tropical)."""
        out = {}
        for i, block in enumerate(self.blocks):
            g = torch.sigmoid(block.attn.score_gate.bias).detach().cpu()
            out[f"block_{i}_attn_gate"] = g.mean().item()
            g_ffn = torch.sigmoid(block.ffn.gate_proj.bias).detach().cpu()
            out[f"block_{i}_ffn_gate"] = g_ffn.mean().item()
        return out
    
    def lf_blend_summary(self) -> dict:
        """Per-block LF blend gate values (1=primal, 0=dual)."""
        out = {}
        for i, block in enumerate(self.blocks):
            lf = block.ffn.lf_act
            if hasattr(lf, "blend_gate"):
                g = torch.sigmoid(lf.blend_gate).detach().cpu().mean().item()
                out[f"block_{i}_lf_gate"] = g
        return out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patchify and embed
        x = self.patchify(x)
        x = self.patch_embed(x)
        
        # Prepend CLS, add positional encoding
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.embed_drop(x + self.pos_embed)
        
        # TropicalBatchNorm at entry
        if self.use_trop_bn:
            x = self.trop_bn(x)
        
        # DTN blocks
        for block in self.blocks:
            x, _ = block(x)
        
        x = self.final_norm(x)
        return self.head(x[:, 0])


# =============================================================================
# Tropical Loss Functions
# =============================================================================

def tropical_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Tropical max-margin loss.
    
    L = max(0, max_{k != y}(logit_k) - logit_y + margin)
    
    This is a max-plus expression:
    - max over incorrect classes is tropical addition
    - subtraction of correct logit is tropical division
    - outer max with 0 is tropical addition with tropical zero
    """
    B = logits.shape[0]
    correct_logits = logits[torch.arange(B, device=logits.device), targets]
    logits_copy = logits.clone()
    logits_copy[torch.arange(B, device=logits.device), targets] = -1e9
    max_wrong = logits_copy.max(dim=1).values
    return F.relu(max_wrong - correct_logits + margin).mean()


# =============================================================================
# Training Utilities
# =============================================================================

def get_mnist_loaders(batch_size: int = 128, data_dir: str = "./data"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def train_epoch(model, loader, optimizer, device, use_tropical_loss=False):
    model.train()
    total_loss = correct = total = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        logits = model(data)
        
        if use_tropical_loss:
            loss = tropical_cross_entropy(logits, target)
        else:
            loss = F.cross_entropy(logits, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        correct += logits.argmax(1).eq(target).sum().item()
        total += data.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = correct = total = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        logits = model(data)
        loss = F.cross_entropy(logits, target)
        total_loss += loss.item() * data.size(0)
        correct += logits.argmax(1).eq(target).sum().item()
        total += data.size(0)
    
    return total_loss / total, correct / total


def main():
    EPOCHS = 25
    BATCH_SIZE = 128
    LR = 1e-3
    D_MODEL = 128
    NUM_LAYERS = 4
    NUM_HEADS = 4
    FFN_DIM = 256
    SAVE_PATH = "deep_tropical_best.pt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    print("=" * 68)
    print("  Deep Tropical Network v2 (Hybrid, TropFormer-based)")
    print("  Tropical-biased hybrid with gated classical fallback")
    print("=" * 68)
    print(f"  Device       : {device}")
    print(f"  d_model      : {D_MODEL}")
    print(f"  num_layers   : {NUM_LAYERS}   num_heads   : {NUM_HEADS}")
    print(f"  ffn_dim      : {FFN_DIM}")
    print(f"  Epochs       : {EPOCHS}    LR          : {LR}")
    
    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)
    print(f"\n  Train : {len(train_loader.dataset):,}   Test : {len(test_loader.dataset):,}")
    
    model = DeepTropNet(
        img_size=28, patch_size=7, in_channels=1, num_classes=10,
        d_model=D_MODEL, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM, init_temp=1.0,
    ).to(device)
    
    print(f"\n  Parameters   : {model.count_params():,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
    print(f"\n{'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  {'TeLoss':>8}  {'TeAcc':>7}  {'Time':>6}")
    print("-" * 55)
    
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        
        flag = " *" if te_acc > best_acc else ""
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), SAVE_PATH)
        
        print(f"{epoch:>3}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {te_loss:>8.4f}  {te_acc:>7.4f}  {elapsed:>5.1f}s{flag}")
    
    print(f"\n  -- Best test accuracy: {best_acc:.4f}  ({best_acc*100:.2f}%) --")
    print(f"  Model saved -> {SAVE_PATH}")
    
    print("\n  Maslov temperatures:")
    for name, temps in model.maslov_summary().items():
        print(f"    {name}: {[f'{t:.3f}' for t in temps.tolist()]}")
    
    print("\n  Gate values (higher = more tropical):")
    for name, val in model.gate_summary().items():
        print(f"    {name}: {val:.3f}")
    
    print("\n  LF blend gates:")
    for name, val in model.lf_blend_summary().items():
        print(f"    {name}: {val:.3f}")
    
    return model, history


if __name__ == "__main__":
    model, history = main()
