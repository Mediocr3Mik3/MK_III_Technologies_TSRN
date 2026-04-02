"""
Classical Transformer Baseline
==============================
A standard transformer with identical hyperparameters to TropFormer,
using only classical linear algebra for comparison.

This serves as the baseline for validating that tropical components
in TropFormer are contributing positively.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =============================================================================
# Classical Multi-Head Attention
# =============================================================================

class ClassicalMultiHeadAttention(nn.Module):
    """
    Standard multi-head attention with classical dot-product scores.
    Matches TropFormer's interface for fair comparison.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self._scale = self.d_k ** -0.5

        # All projections are classical linear
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape

        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))

        # Classical dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self._scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out), attn


# =============================================================================
# Classical FFN
# =============================================================================

class ClassicalFFN(nn.Module):
    """
    Standard feed-forward network with GELU activation.
    Matches TropFormer's FFN dimensions.
    """

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.up_proj = nn.Linear(d_model, ffn_dim)
        self.gelu = nn.GELU()
        self.down_proj = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.gelu(self.up_proj(x))
        out = self.dropout(self.down_proj(out))
        return self.norm(out + x)


# =============================================================================
# Classical Transformer Block
# =============================================================================

class ClassicalTransformerBlock(nn.Module):
    """
    Standard transformer block with pre-LayerNorm (same as TropFormer).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ClassicalMultiHeadAttention(d_model, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ffn = ClassicalFFN(d_model, ffn_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_w = self.attn(self.norm1(x), mask)
        x = x + self.drop1(attn_out)
        x = self.ffn(x)
        return x, attn_w


# =============================================================================
# Classical Transformer (Vision)
# =============================================================================

class ClassicalTransformer(nn.Module):
    """
    Classical Vision Transformer for image classification.
    Identical architecture to TropFormer but using only classical components.
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        num_classes: int = 10,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, d_model)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)

        # CLS token and positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.embed_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ClassicalTransformerBlock(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        x = self.patchify(x)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.embed_drop(x + self.pos_embed)

        for block in self.blocks:
            x, _ = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])


# =============================================================================
# Data Loading
# =============================================================================

def get_mnist_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train, shuffle=True, **kw),
        DataLoader(test, shuffle=False, **kw),
    )


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(model, loader, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = correct = total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(data)
                loss = F.cross_entropy(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        bs = data.size(0)
        total_loss += loss.item() * bs
        correct += logits.argmax(1).eq(target).sum().item()
        total += bs

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = correct = total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        logits = model(data)
        loss = F.cross_entropy(logits, target)
        bs = data.size(0)
        total_loss += loss.item() * bs
        correct += logits.argmax(1).eq(target).sum().item()
        total += bs

    return total_loss / total, correct / total


# =============================================================================
# Main
# =============================================================================

def main():
    # Hyperparameters (identical to TropFormer)
    EPOCHS = 25
    BATCH_SIZE = 128
    LR = 3e-3
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    FFN_DIM = 256
    DROPOUT = 0.1
    DATA_DIR = "./data"
    SAVE_PATH = "classical_transformer_best.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    print("=" * 68)
    print("  Classical Transformer Baseline")
    print("  Standard dot-product attention - GELU FFN")
    print("=" * 68)
    print(f"  Device       : {device}")
    print(f"  d_model      : {D_MODEL}   num_heads : {NUM_HEADS}")
    print(f"  num_layers   : {NUM_LAYERS}   ffn_dim   : {FFN_DIM}")
    print(f"  Epochs       : {EPOCHS}    LR        : {LR}")

    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE, DATA_DIR)
    print(f"\n  Train : {len(train_loader.dataset):,}   "
          f"Test  : {len(test_loader.dataset):,}")

    model = ClassicalTransformer(
        img_size=28, patch_size=7, in_channels=1, num_classes=10,
        d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM, dropout=DROPOUT,
    ).to(device)

    print(f"\n  Parameters   : {model.count_params():,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    total_steps = EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos",
    )
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    best_acc = 0.0
    history = {k: [] for k in ("train_loss", "train_acc", "test_loss", "test_acc")}

    header = (f"\n{'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}"
              f"  {'TeLoss':>8}  {'TeAcc':>7}  {'Time':>6}")
    print(header)
    print("-" * len(header.strip()))

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, scaler
        )
        te_loss, te_acc = evaluate(model, test_loader, device)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        flag = " *" if te_acc > best_acc else ""
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), SAVE_PATH)

        print(f"{epoch:>3}  {tr_loss:>8.4f}  {tr_acc:>7.4f}"
              f"  {te_loss:>8.4f}  {te_acc:>7.4f}  {elapsed:>5.1f}s{flag}")

    print(f"\n  -- Best test accuracy: {best_acc:.4f}  ({best_acc*100:.2f}%) --")
    print(f"  Model saved -> {SAVE_PATH}")

    return model, history


if __name__ == "__main__":
    model, history = main()
