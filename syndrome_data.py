"""
Toric code syndrome data generation for ML decoding benchmarks.

Uses the toric code (periodic boundary conditions) for clean geometry.
Bit-flip noise model (code capacity): each qubit independently flips with prob p.
Plaquette (X) stabilizers detect Z errors.
Task: predict logical observable (parity of Z errors on non-contractible cycle).

Standard benchmark: MWPM threshold ~10.3% for bit-flip code capacity.
"""

import numpy as np
import torch
from typing import Tuple, Dict, List


# ---------------------------------------------------------------------------
#  Toric code geometry
# ---------------------------------------------------------------------------

class ToricCode:
    """Toric code on a d x d lattice with periodic boundary conditions.
    
    Data qubits live on edges:
      - Horizontal edge (r, c): index = r*d + c           (0 .. d^2-1)
      - Vertical edge (r, c):   index = d^2 + r*d + c     (d^2 .. 2d^2-1)
    Total: 2*d^2 data qubits.
    
    Vertex (star) operators detect Z errors. Vertex (r, c) touches:
      - h(r, c):       horizontal edge to the right
      - h(r, (c-1)%d): horizontal edge to the left
      - v(r, c):       vertical edge going down
      - v((r-1)%d, c): vertical edge coming from above
    
    Logical Z operator (first logical qubit):
      Z on all horizontal edges in row 0: {h(0,c) for c in range(d)}
      (non-contractible horizontal cycle, commutes with all vertex operators)
    """
    
    def __init__(self, d: int):
        self.d = d
        self.n_qubits = 2 * d * d
        self.n_vertices = d * d
        
        # Build vertex -> adjacent qubit mapping
        self.vert_qubits = []  # vert_qubits[i] = list of 4 qubit indices
        for r in range(d):
            for c in range(d):
                h_right = r * d + c                        # h(r, c)
                h_left  = r * d + (c - 1) % d              # h(r, (c-1)%d)
                v_down  = d * d + r * d + c                # v(r, c)
                v_up    = d * d + ((r - 1) % d) * d + c    # v((r-1)%d, c)
                self.vert_qubits.append([h_right, h_left, v_down, v_up])
        
        # Build parity check matrix H: (n_vertices x n_qubits)
        self.H = np.zeros((self.n_vertices, self.n_qubits), dtype=np.uint8)
        for i, qubits in enumerate(self.vert_qubits):
            for q in qubits:
                self.H[i, q] = 1
        
        # Logical Z operator: horizontal edges in row 0
        self.logical_z = np.zeros(self.n_qubits, dtype=np.uint8)
        for c in range(d):
            self.logical_z[c] = 1  # h(0, c) = 0*d + c
    
    def syndrome(self, errors: np.ndarray) -> np.ndarray:
        """Compute vertex syndrome from Z-error pattern. errors: (..., n_qubits)."""
        return (errors @ self.H.T) % 2
    
    def logical_value(self, errors: np.ndarray) -> np.ndarray:
        """Compute logical observable value. errors: (..., n_qubits)."""
        return (errors @ self.logical_z) % 2
    
    def verify(self):
        """Verify code properties."""
        d = self.d
        # 1. Each vertex has exactly 4 adjacent qubits
        for i, qs in enumerate(self.vert_qubits):
            assert len(qs) == 4, f"Vertex {i} has {len(qs)} qubits"
        
        # 2. Logical operator has syndrome 0 (commutes with all vertex operators)
        s = (self.H @ self.logical_z) % 2
        assert np.all(s == 0), "Logical Z doesn't commute with vertex operators!"
        
        # 3. Single-qubit error produces syndrome with exactly 2 defects
        for q in range(self.n_qubits):
            e = np.zeros(self.n_qubits, dtype=np.uint8)
            e[q] = 1
            s = (self.H @ e) % 2
            assert s.sum() == 2, f"Qubit {q}: syndrome weight {s.sum()}, expected 2"
        
        # 4. Minimum weight logical has weight d
        assert self.logical_z.sum() == d
        
        print(f"  ToricCode d={d}: {self.n_qubits} qubits, "
              f"{self.n_vertices} vertices — all checks passed")


# ---------------------------------------------------------------------------
#  Data generation
# ---------------------------------------------------------------------------

def generate_syndrome_data(d: int, p: float, n_samples: int,
                           rng: np.random.Generator
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (syndrome, logical_label) pairs for toric code bit-flip noise.
    
    Returns:
        syndromes: (n_samples, d^2) uint8 array
        labels: (n_samples,) uint8 array (0 or 1)
    """
    code = ToricCode(d)
    errors = rng.random((n_samples, code.n_qubits)) < p
    errors = errors.astype(np.uint8)
    
    syndromes = code.syndrome(errors)
    labels = code.logical_value(errors)
    
    return syndromes, labels


# ---------------------------------------------------------------------------
#  Dataset class for TSRN training
# ---------------------------------------------------------------------------

class SyndromeDataset:
    """Toric code syndrome dataset for TSRN autoregressive training.
    
    Sequence format: [s0 s1 ... sN SEP]  ->  [s1 ... sN SEP label]
    Vocab: {0, 1, SEP=2}
    
    At inference, the model sees all syndrome bits + SEP and predicts label.
    """
    
    VOCAB_SIZE = 3  # 0, 1, SEP
    SEP_TOKEN = 2
    
    def __init__(self, d: int, p_train: float = 0.05,
                 n_train: int = 200_000, n_val: int = 20_000, n_test: int = 20_000,
                 context_len: int = 64, seed: int = 42):
        self.d = d
        self.p_train = p_train
        self.context_len = context_len
        self.vocab_sz = self.VOCAB_SIZE
        self.ctx = context_len
        
        code = ToricCode(d)
        code.verify()
        self.code = code
        
        rng = np.random.default_rng(seed)
        n_synd = code.n_vertices  # d^2 syndrome bits
        seq_len = n_synd + 1  # syndrome + SEP
        
        print(f"\n  Generating toric code syndrome data (d={d}, p={p_train})")
        print(f"  Syndrome bits: {n_synd}, Sequence length: {seq_len}+1")
        
        # Generate splits
        splits = {}
        for name, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
            synds, labels = generate_syndrome_data(d, p_train, n, rng)
            
            # Build sequences: [syndrome_bits SEP] and targets [syndrome_bits[1:] SEP label]
            x = np.zeros((n, seq_len), dtype=np.int64)
            y = np.zeros((n, seq_len), dtype=np.int64)
            
            x[:, :n_synd] = synds
            x[:, n_synd] = self.SEP_TOKEN
            
            y[:, :n_synd - 1] = synds[:, 1:]
            y[:, n_synd - 1] = self.SEP_TOKEN
            y[:, n_synd] = labels
            
            splits[name] = (torch.tensor(x), torch.tensor(y))
            err_rate = labels.mean()
            print(f"    {name}: {n:,} samples, logical_err_rate={err_rate:.4f}")
        
        self.train_x, self.train_y = splits["train"]
        self.val_x, self.val_y = splits["val"]
        self.test_x, self.test_y = splits["test"]
        self.seq_len = seq_len
        self.n_synd = n_synd
        
        print(f"  Vocab: {self.VOCAB_SIZE}  Context: {seq_len}")
    
    def batch(self, split: str, B: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random batch of (input, target) sequences."""
        if split == "test":
            dx, dy = self.test_x, self.test_y
        elif split == "val":
            dx, dy = self.val_x, self.val_y
        else:
            dx, dy = self.train_x, self.train_y
        
        ix = torch.randint(len(dx), (B,))
        return dx[ix].to(device), dy[ix].to(device)
    
    def get_split(self, split: str):
        """Get full split tensors."""
        if split == "test":
            return self.test_x, self.test_y
        elif split == "val":
            return self.val_x, self.val_y
        return self.train_x, self.train_y


# ---------------------------------------------------------------------------
#  Evaluation utilities
# ---------------------------------------------------------------------------

def evaluate_decoder_accuracy(model, dataset: SyndromeDataset,
                              device, split: str = "test",
                              batch_size: int = 256) -> Dict:
    """Evaluate TSRN decoder accuracy on syndrome classification.
    
    Returns dict with accuracy, logical_error_rate, and per-class stats.
    """
    model.eval()
    x_all, y_all = dataset.get_split(split)
    n = len(x_all)
    label_pos = dataset.n_synd  # position in target where label lives
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = x_all[start:end].to(device)
            y = y_all[start:end].to(device)
            
            logits, _ = model(x)  # (B, T, V)
            
            # Get prediction at label position
            pred_logits = logits[:, label_pos, :2]  # only tokens 0 and 1
            preds = pred_logits.argmax(dim=-1)
            labels = y[:, label_pos]
            
            correct += (preds == labels).sum().item()
            total += len(x)
    
    accuracy = correct / total
    logical_error_rate = 1.0 - accuracy
    
    return {
        'accuracy': accuracy,
        'logical_error_rate': logical_error_rate,
        'n_samples': total,
    }


def evaluate_at_error_rates(model, code_d: int, device,
                            error_rates: List[float] = None,
                            n_samples: int = 10_000,
                            batch_size: int = 256,
                            seed: int = 999) -> Dict:
    """Evaluate trained model at multiple physical error rates.
    
    Generates fresh test data at each error rate and measures logical error rate.
    """
    if error_rates is None:
        error_rates = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    
    model.eval()
    code = ToricCode(code_d)
    results = {}
    
    for p in error_rates:
        rng = np.random.default_rng(seed)
        synds, labels = generate_syndrome_data(code_d, p, n_samples, rng)
        
        # Build input sequences
        n_synd = code.n_vertices
        x = np.zeros((n_samples, n_synd + 1), dtype=np.int64)
        x[:, :n_synd] = synds
        x[:, n_synd] = SyndromeDataset.SEP_TOKEN
        x_t = torch.tensor(x)
        
        correct = 0
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                xb = x_t[start:end].to(device)
                logits, _ = model(xb)
                pred_logits = logits[:, n_synd, :2]
                preds = pred_logits.argmax(dim=-1).cpu().numpy()
                correct += (preds == labels[start:end]).sum()
        
        accuracy = correct / n_samples
        logical_err = 1.0 - accuracy
        raw_err = labels.mean()
        
        results[p] = {
            'physical_error_rate': p,
            'logical_error_rate': float(logical_err),
            'raw_logical_rate': float(raw_err),
            'accuracy': float(accuracy),
            'n_samples': n_samples,
        }
    
    return results


# ---------------------------------------------------------------------------
#  MWPM baseline decoder
# ---------------------------------------------------------------------------

def _torus_manhattan(p1, p2, d):
    """Manhattan distance on d x d torus."""
    dr = min(abs(p1[0] - p2[0]), d - abs(p1[0] - p2[0]))
    dc = min(abs(p1[1] - p2[1]), d - abs(p1[1] - p2[1]))
    return dr + dc


def _min_weight_perfect_matching(positions, d):
    """Brute-force minimum weight perfect matching for defect positions.
    
    Exact for small defect counts (typical for d<=7 at low error rates).
    Returns list of (i, j) index pairs.
    """
    n = len(positions)
    if n == 0:
        return []
    if n == 2:
        return [(0, 1)]

    best = [float('inf'), None]  # [weight, matching]

    def _search(remaining, matching, weight):
        if not remaining:
            if weight < best[0]:
                best[0] = weight
                best[1] = list(matching)
            return
        if weight >= best[0]:
            return  # prune
        first = remaining[0]
        for idx in range(1, len(remaining)):
            partner = remaining[idx]
            w = _torus_manhattan(positions[first], positions[partner], d)
            new_rem = remaining[1:idx] + remaining[idx+1:]
            matching.append((first, partner))
            _search(new_rem, matching, weight + w)
            matching.pop()

    _search(list(range(n)), [], 0)
    return best[1]


def _correction_logical_parity(pos_i, pos_j, d):
    """Parity of logical-Z edges crossed by canonical correction path.
    
    Canonical path: horizontal at row of pos_i (shortest direction on torus),
    then vertical at destination column.
    Logical Z = horizontal edges at row 0.
    """
    r1, c1 = pos_i
    r2, c2 = pos_j
    # Shortest horizontal displacement
    dc_right = (c2 - c1) % d
    dc_left = (c1 - c2) % d
    h_steps = min(dc_right, dc_left)
    # Only contributes if horizontal leg is at row 0
    if r1 == 0:
        return h_steps % 2
    return 0


def mwpm_decode_single(syndrome, d):
    """Decode a single syndrome using MWPM. Returns predicted logical (0 or 1)."""
    defects = np.where(syndrome == 1)[0]
    n = len(defects)
    if n == 0:
        return 0
    if n % 2 != 0:
        return 0  # invalid syndrome

    positions = [(idx // d, idx % d) for idx in defects]
    matching = _min_weight_perfect_matching(positions, d)

    logical = 0
    for i, j in matching:
        logical ^= _correction_logical_parity(positions[i], positions[j], d)
    return logical


def mwpm_evaluate(d, p, n_samples=10000, seed=999):
    """Evaluate MWPM decoder at a given physical error rate.
    
    Returns dict with accuracy, logical_error_rate, raw_logical_rate.
    """
    rng = np.random.default_rng(seed)
    syndromes, labels = generate_syndrome_data(d, p, n_samples, rng)

    correct = 0
    for i in range(n_samples):
        pred = mwpm_decode_single(syndromes[i], d)
        if pred == labels[i]:
            correct += 1

    accuracy = correct / n_samples
    raw_err = float(labels.mean())
    return {
        'accuracy': accuracy,
        'logical_error_rate': 1.0 - accuracy,
        'raw_logical_rate': raw_err,
        'n_samples': n_samples,
    }


def mwpm_sweep(d, error_rates=None, n_samples=10000, seed=999):
    """Run MWPM at multiple error rates for a given code distance."""
    if error_rates is None:
        error_rates = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    results = {}
    for p in error_rates:
        results[p] = mwpm_evaluate(d, p, n_samples, seed)
    return results


# ---------------------------------------------------------------------------
#  Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Verifying toric codes...")
    for d in [3, 5, 7]:
        code = ToricCode(d)
        code.verify()
    
    print("\nGenerating test data (d=5, p=0.05, 1000 samples)...")
    rng = np.random.default_rng(42)
    synds, labels = generate_syndrome_data(5, 0.05, 1000, rng)
    print(f"  Syndrome shape: {synds.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Logical error rate: {labels.mean():.4f}")
    print(f"  Mean syndrome weight: {synds.sum(axis=1).mean():.1f}")
    
    print("\nBuilding SyndromeDataset (d=5, p=0.05)...")
    ds = SyndromeDataset(d=5, p_train=0.05, n_train=1000, n_val=200, n_test=200)
    x, y = ds.batch("train", 4, "cpu")
    print(f"  Batch x shape: {x.shape}, y shape: {y.shape}")
    print(f"  Sample x: {x[0].tolist()}")
    print(f"  Sample y: {y[0].tolist()}")
    
    print("\nDone!")
