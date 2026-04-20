"""
Tropical Path-Planning Optimizers for TSRN
==========================================

Six-phase implementation of tropical optimization for the TSRN architecture.

Phase 1: TropicalSubgradientOptimizer
    - Replaces classical gradients with tropical subgradients for max-plus layers
    - Sparse by construction: only the "winning path" contributes

Phase 2: TropicalMirrorDescent
    - Mirror descent on the tropical polytope
    - Bregman divergence w.r.t. tropical entropy

Phase 3: TropicalPathOptimizer
    - Shortest-path optimization on the loss landscape's tropical dual graph
    - Local cell enumeration via activation pattern recording

Phase 4: TropicalGeodesicLR
    - Learning rate scheduler derived from tropical geodesic distance
    - Adapts to the tropical geometry of the loss landscape

Phase 5: MaslovDequantizationSchedule
    - Per-module tropicalization via temperature annealing
    - Classical (ε=1) → tropical (ε→0) over training

Phase 6: TropicalRGFlowOptimizer
    - Multi-scale optimization via RG coarse-graining in parameter space
    - Coarsen → optimize → refine → polish

Usage:
    from tropical_optimizers import (
        TropicalSubgradientOptimizer,
        TropicalMirrorDescent,
        TropicalPathOptimizer,
        TropicalGeodesicLR,
        MaslovDequantizationSchedule,
        TropicalRGFlowOptimizer,
        TropicalHybridTrainer,
    )
"""

import math
import copy
from typing import Dict, List, Optional, Tuple, Set, Callable, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1: Tropical Subgradient Optimizer
# ═══════════════════════════════════════════════════════════════════════════

class TropicalSubgradientOptimizer(Optimizer):
    """Tropical subgradient descent for max-plus network layers.

    For a tropical polynomial f(x) = max_i(a_i + <w_i, x>), the tropical
    subgradient at x is w_{i*} where i* = argmax_i(a_i + <w_i, x>).

    This optimizer:
    1. Uses tropical subgradients for designated "tropical" parameter groups
    2. Falls back to standard AdamW for "classical" parameter groups
    3. Supports stochastic tie-breaking when multiple linear pieces tie

    Mathematical basis:
        ∂_T f(x) = conv{w_i : i ∈ argmax_j(a_j + <w_j, x>)}

    The subgradient is the convex hull of the weight vectors of all
    maximally active linear pieces. We use a stochastic selection from
    this set for the update direction.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, tropical_momentum=0.9,
                 tie_threshold=1e-6):
        """
        Args:
            params: iterable of parameters or dicts. Each dict may contain
                    'tropical': bool to indicate tropical subgradient treatment.
            lr: learning rate
            betas: AdamW betas for classical parameters
            eps: AdamW epsilon
            weight_decay: L2 regularization
            tropical_momentum: momentum for tropical subgradient updates
            tie_threshold: threshold for detecting tied argmax values
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        tropical_momentum=tropical_momentum,
                        tie_threshold=tie_threshold, tropical=False)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Tropical parameters: momentum-based subgradient update
        Classical parameters: AdamW update
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            is_tropical = group.get('tropical', False)
            lr = group['lr']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if wd != 0:
                    p.mul_(1 - lr * wd)

                if is_tropical:
                    # Tropical subgradient update with momentum
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['momentum_buffer'] = torch.zeros_like(p)
                        state['sparsity'] = 0.0

                    state['step'] += 1
                    mu = group['tropical_momentum']
                    buf = state['momentum_buffer']

                    # Sparsify the gradient: only update along the
                    # "winning path" directions (non-zero gradient entries
                    # from the tropical max operation)
                    sparse_grad = self._sparsify_tropical_grad(
                        grad, group['tie_threshold'])

                    # Track sparsity for diagnostics
                    total = grad.numel()
                    nonzero = (sparse_grad != 0).sum().item()
                    state['sparsity'] = 1.0 - nonzero / max(total, 1)

                    # Momentum update
                    buf.mul_(mu).add_(sparse_grad)
                    p.add_(buf, alpha=-lr)

                else:
                    # AdamW update for classical parameters
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    state['step'] += 1
                    beta1, beta2 = group['betas']
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    step_size = lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                             ).add_(group['eps'])

                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    @staticmethod
    def _sparsify_tropical_grad(grad: Tensor, threshold: float) -> Tensor:
        """Extract the tropical subgradient from the classical gradient.

        The tropical subgradient is sparse — only the coordinates
        corresponding to the active linear piece(s) of the tropical
        polynomial contribute.

        For gradients from max operations, non-zero entries correspond
        to the winning path. We preserve these and zero out the rest.
        """
        # The gradient from max/logsumexp operations is naturally sparse
        # (only the argmax contributes in the limit). We enhance this
        # by zeroing entries below a relative threshold.
        abs_grad = grad.abs()
        max_val = abs_grad.max()
        if max_val == 0:
            return grad
        # Keep only entries within threshold of the maximum gradient magnitude
        mask = abs_grad >= max_val * threshold
        return grad * mask.float()

    def get_tropical_sparsity(self) -> Dict[str, float]:
        """Return sparsity stats for tropical parameter groups."""
        stats = {}
        for i, group in enumerate(self.param_groups):
            if group.get('tropical', False):
                sparsities = []
                for p in group['params']:
                    if p in self.state and 'sparsity' in self.state[p]:
                        sparsities.append(self.state[p]['sparsity'])
                if sparsities:
                    stats[f'group_{i}'] = {
                        'mean_sparsity': sum(sparsities) / len(sparsities),
                        'min_sparsity': min(sparsities),
                        'max_sparsity': max(sparsities),
                    }
        return stats


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2: Tropical Mirror Descent
# ═══════════════════════════════════════════════════════════════════════════

class TropicalMirrorDescent(Optimizer):
    """Mirror descent on the tropical polytope.

    Uses the Bregman divergence with respect to the tropical entropy:
        φ(θ) = max_i θ_i - (1/n) Σ_i θ_i

    The mirror map:
        ψ = ∇φ(θ)  →  update in dual space  →  θ = (∇φ)^{-1}(ψ)

    For tropical parameters, the update rule is:
        1. Dual step:    ψ_t = ∇φ(θ_t) - lr · g_t
        2. Mirror step:  θ_{t+1} = ∇φ*(ψ_t)
        3. Project:      θ_{t+1} = Π_Δ(θ_{t+1})

    where Δ is the tropical simplex {θ : max_i θ_i = 0} and
    g_t is the tropical subgradient.

    Mathematical basis:
        D_φ(θ, θ') = φ(θ) - φ(θ') - <∇φ(θ'), θ - θ'>

    The tropical entropy φ measures the "spread" of parameter values.
    Its Bregman divergence penalizes parameter updates that increase
    the gap between the largest and smallest parameter coordinates.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, tropical_lr=1e-2,
                 project_to_simplex=True):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        tropical_lr=tropical_lr,
                        project_to_simplex=project_to_simplex,
                        tropical=False)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            is_tropical = group.get('tropical', False)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                wd = group['weight_decay']

                if wd != 0:
                    p.mul_(1 - group['lr'] * wd)

                if is_tropical:
                    lr_t = group['tropical_lr']
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['dual_var'] = self._tropical_grad_phi(p.data.clone())

                    state['step'] += 1

                    # 1. Dual step: ψ ← ψ - lr · g
                    state['dual_var'].sub_(grad, alpha=lr_t)

                    # 2. Mirror step: θ ← (∇φ*)^{-1}(ψ)
                    # For tropical entropy, the inverse mirror map is
                    # approximately the identity shifted by the mean
                    new_theta = self._inverse_tropical_mirror(state['dual_var'])

                    # 3. Project onto tropical simplex if needed
                    if group['project_to_simplex']:
                        new_theta = self._project_tropical_simplex(new_theta)

                    p.data.copy_(new_theta)

                else:
                    # Standard AdamW for classical params
                    lr = group['lr']
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    state['step'] += 1
                    beta1, beta2 = group['betas']
                    state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(
                        grad, grad, value=1 - beta2)

                    bc1 = 1 - beta1 ** state['step']
                    bc2 = 1 - beta2 ** state['step']
                    step_size = lr / bc1
                    denom = (state['exp_avg_sq'].sqrt() /
                             math.sqrt(bc2)).add_(group['eps'])
                    p.addcdiv_(state['exp_avg'], denom, value=-step_size)

        return loss

    @staticmethod
    def _tropical_grad_phi(theta: Tensor) -> Tensor:
        """Gradient of tropical entropy φ(θ) = max(θ) - mean(θ).

        ∇φ(θ)_i = (1_{i = argmax} - 1/n)

        Returns a tensor in the dual space.
        """
        n = theta.numel()
        grad = torch.full_like(theta, -1.0 / n)
        flat = theta.view(-1)
        max_idx = flat.argmax()
        grad.view(-1)[max_idx] += 1.0
        return grad

    @staticmethod
    def _inverse_tropical_mirror(psi: Tensor) -> Tensor:
        """Inverse mirror map for tropical entropy.

        Approximate: shift so that the max entry maps to 0,
        then rescale to match the original parameter magnitude.
        """
        return psi - psi.max()

    @staticmethod
    def _project_tropical_simplex(theta: Tensor) -> Tensor:
        """Project onto tropical simplex Δ_T = {θ : max_i θ_i = 0}.

        This is simply θ → θ - max(θ).
        """
        return theta - theta.max()


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3: Tropical Path Optimizer
# ═══════════════════════════════════════════════════════════════════════════

class ActivationPatternRecorder:
    """Records which linear regions (cells) are active during forward pass.

    For tropical (max) operations, the activation pattern is the index
    of the winning linear piece at each position. For ReLU-like activations,
    it's the sign pattern. Together, these define a cell in the tropical
    cell decomposition of parameter space.
    """

    def __init__(self):
        self.patterns: Dict[str, Tensor] = {}
        self._hooks: List[Any] = []

    def register_hooks(self, model: nn.Module):
        """Register forward hooks on all max/ReLU-like operations."""
        for name, module in model.named_modules():
            if self._is_tropical_module(module):
                hook = module.register_forward_hook(
                    self._make_hook(name))
                self._hooks.append(hook)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_cell_code(self) -> Tuple[int, ...]:
        """Return a hashable cell identifier from current activation patterns."""
        code = []
        for name in sorted(self.patterns.keys()):
            # Hash the activation pattern to a compact integer
            pattern = self.patterns[name]
            code.append(hash(pattern.cpu().numpy().tobytes()))
        return tuple(code)

    def clear(self):
        self.patterns.clear()

    @staticmethod
    def _is_tropical_module(module: nn.Module) -> bool:
        """Check if a module contains tropical (max-plus) operations."""
        cls_name = type(module).__name__
        return cls_name in ('TropicalAttention', 'TropicalSSM', 'RGPool')

    def _make_hook(self, name: str):
        def hook(module, input, output):
            # Record which "path" won in the tropical computation
            if isinstance(output, Tensor):
                # Binary pattern: sign of the output (which linear piece)
                self.patterns[name] = (output > 0).byte()
            elif isinstance(output, tuple):
                self.patterns[name] = (output[0] > 0).byte()
        return hook


class TropicalPathOptimizer:
    """Shortest-path optimization on the tropical dual graph.

    The loss landscape of a PWL (piecewise-linear) network is a tropical
    hypersurface. The dual graph has:
    - Nodes: linear regions (cells) of the loss function
    - Edges: adjacent cells (sharing a codimension-1 face)

    We approximate this by:
    1. Recording activation patterns to identify the current cell
    2. Enumerating neighbor cells by flipping activation bits
    3. Evaluating loss in promising neighbor cells
    4. Moving toward the best neighbor's centroid
    5. Fine-tuning with AdamW within the new cell

    This is a tropical analogue of the Nelder-Mead simplex method,
    but guided by the cell structure of the tropical variety.
    """

    def __init__(self, model: nn.Module, base_optimizer: Optimizer,
                 n_neighbors: int = 5, exploration_lr: float = 0.01,
                 path_budget: int = 3):
        """
        Args:
            model: the neural network
            base_optimizer: AdamW or similar for within-cell fine-tuning
            n_neighbors: number of neighbor cells to evaluate per step
            exploration_lr: step size for cell-boundary exploration
            path_budget: max forward evaluations for neighbor search
        """
        self.model = model
        self.base_optimizer = base_optimizer
        self.n_neighbors = n_neighbors
        self.exploration_lr = exploration_lr
        self.path_budget = path_budget
        self.recorder = ActivationPatternRecorder()
        self.cell_history: List[Tuple[Tuple, float]] = []
        self._best_loss = float('inf')
        self._best_params = None

    def step(self, loss_fn: Callable, x: Tensor, y: Tensor):
        """One tropical path optimization step.

        1. Record current cell
        2. Evaluate neighbors
        3. Move to best cell
        4. Fine-tune within cell
        """
        # Step 1: Record current cell and loss
        self.recorder.register_hooks(self.model)
        with torch.no_grad():
            _, current_loss = loss_fn(x, y)
        current_cell = self.recorder.get_cell_code()
        self.cell_history.append((current_cell, current_loss.item()))
        self.recorder.remove_hooks()

        # Track best
        if current_loss.item() < self._best_loss:
            self._best_loss = current_loss.item()
            self._best_params = {k: v.clone()
                                 for k, v in self.model.state_dict().items()}

        # Step 2: Explore neighbor cells by perturbing parameters
        best_neighbor_loss = current_loss.item()
        best_perturbation = None

        saved_state = {k: v.clone()
                       for k, v in self.model.state_dict().items()}

        for _ in range(min(self.n_neighbors, self.path_budget)):
            # Generate a random perturbation direction
            perturbation = {}
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    # Perturbation along gradient direction + random noise
                    direction = torch.randn_like(p)
                    if p.grad is not None:
                        # Bias toward gradient descent direction
                        direction = 0.5 * direction + 0.5 * (-p.grad)
                    direction = direction / (direction.norm() + 1e-8)
                    perturbation[name] = direction

            # Apply perturbation
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name in perturbation:
                        p.add_(perturbation[name], alpha=self.exploration_lr)

            # Evaluate
            with torch.no_grad():
                _, neighbor_loss = loss_fn(x, y)

            if neighbor_loss.item() < best_neighbor_loss:
                best_neighbor_loss = neighbor_loss.item()
                best_perturbation = {k: v.clone()
                                     for k, v in perturbation.items()}

            # Restore
            self.model.load_state_dict(saved_state)

        # Step 3: Move to best neighbor if better than current
        if best_perturbation is not None and best_neighbor_loss < current_loss.item():
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name in best_perturbation:
                        p.add_(best_perturbation[name],
                               alpha=self.exploration_lr)

        # Step 4: Fine-tune within cell using base optimizer
        self.base_optimizer.zero_grad(set_to_none=True)
        _, loss = loss_fn(x, y)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.base_optimizer.step()

        return loss.item(), best_neighbor_loss

    def get_path_stats(self) -> Dict:
        """Return statistics about the optimization path."""
        if not self.cell_history:
            return {}
        cells_visited = len(set(c for c, _ in self.cell_history))
        losses = [l for _, l in self.cell_history]
        return {
            'cells_visited': cells_visited,
            'total_steps': len(self.cell_history),
            'cell_diversity': cells_visited / max(len(self.cell_history), 1),
            'best_loss': min(losses),
            'current_loss': losses[-1] if losses else float('inf'),
            'loss_improvement': losses[0] - losses[-1] if len(losses) > 1 else 0,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 4: Tropical Geodesic Learning Rate Scheduler
# ═══════════════════════════════════════════════════════════════════════════

class TropicalGeodesicLR(_LRScheduler):
    """Learning rate scheduler based on tropical geodesic distance.

    The tropical metric on ℝ^n/ℝ·1 is:
        d_T(x, y) = max_i(x_i - y_i) - min_i(x_i - y_i)

    We approximate the distance to the nearest local minimum using
    the tropical "spread" of the gradient:
        d_T ≈ max_i(g_i) - min_i(g_i)

    The learning rate decays proportionally to this distance:
        lr_t = lr_base · d_T(g_t) / d_T(g_0)

    This naturally adapts:
    - Far from minimum → large gradient spread → large lr
    - Near minimum → gradients agree → small lr
    - At critical point → zero spread → zero lr

    The schedule also incorporates warmup and a minimum lr floor.
    """

    def __init__(self, optimizer: Optimizer, model: nn.Module,
                 warmup_steps: int = 1000, total_steps: int = 100000,
                 lr_min_ratio: float = 0.1, smoothing: float = 0.99,
                 last_epoch: int = -1):
        self.model = model
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_min_ratio = lr_min_ratio
        self.smoothing = smoothing

        # EMA of tropical distance
        self._tropical_dist_ema = None
        self._initial_dist = None
        self._step_count = 0

        super().__init__(optimizer, last_epoch)

    def _compute_tropical_distance(self) -> float:
        """Compute tropical distance from gradient statistics.

        d_T(g) = max(g_flat) - min(g_flat) across all parameters.
        This measures how much the gradients "disagree" — a proxy
        for distance to a critical point.
        """
        all_grads = []
        for p in self.model.parameters():
            if p.grad is not None and p.grad.numel() > 0:
                flat = p.grad.detach().view(-1)
                all_grads.append(flat)

        if not all_grads:
            return 1.0

        combined = torch.cat(all_grads)
        # Tropical distance: max - min of gradient coordinates
        d_t = (combined.max() - combined.min()).item()
        return max(d_t, 1e-10)

    def step_with_tropical_distance(self, tropical_dist: Optional[float] = None):
        """Call this instead of step() to also update the tropical distance."""
        if tropical_dist is None:
            tropical_dist = self._compute_tropical_distance()

        self._step_count += 1

        if self._tropical_dist_ema is None:
            self._tropical_dist_ema = tropical_dist
            self._initial_dist = tropical_dist
        else:
            alpha = self.smoothing
            self._tropical_dist_ema = (
                alpha * self._tropical_dist_ema + (1 - alpha) * tropical_dist)

        super().step()

    def get_lr(self):
        """Compute learning rate based on tropical geodesic distance."""
        step = self._step_count

        # Warmup phase
        if step < self.warmup_steps:
            warmup_factor = step / max(self.warmup_steps, 1)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # Post-warmup: tropical geodesic decay
        if self._initial_dist is None or self._initial_dist == 0:
            ratio = 1.0
        else:
            ratio = (self._tropical_dist_ema or 1.0) / self._initial_dist
            ratio = max(ratio, self.lr_min_ratio)  # floor

        # Cosine modulation for smoothness
        progress = (step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        # Blend: tropical distance provides the magnitude,
        # cosine provides smooth decay envelope
        blended = 0.7 * ratio + 0.3 * cosine_factor
        blended = max(blended, self.lr_min_ratio)

        return [base_lr * blended for base_lr in self.base_lrs]


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 5: Maslov Dequantization Schedule
# ═══════════════════════════════════════════════════════════════════════════

class MaslovDequantizationSchedule:
    """Per-module tropicalization via Maslov dequantization.

    The Maslov dequantization parameter ε controls the classical↔tropical
    interpolation:
        f_ε(x) = ε · log(Σ_i exp(f_i(x)/ε))

    As ε → 0:  f_ε → max_i f_i(x)  (pure tropical)
    At ε = 1:  f_ε = logsumexp(f_i)  (standard softmax/classical)

    This schedule anneals ε from 1.0 (classical) to near 0 (tropical)
    over the course of training, with per-module control.

    The schedule:
        Phase 1 (warmup):     ε = 1.0           (fully classical)
        Phase 2 (anneal):     ε: 1.0 → 0.1      (gradually tropical)
        Phase 3 (stabilize):  ε = 0.1            (mostly tropical)
        Phase 4 (final):      ε: 0.1 → ε_min    (near-pure tropical)
    """

    def __init__(self, total_steps: int, epsilon_min: float = 0.01,
                 warmup_frac: float = 0.1, anneal_frac: float = 0.4,
                 stabilize_frac: float = 0.3):
        """
        Args:
            total_steps: total training steps
            epsilon_min: final tropical temperature
            warmup_frac: fraction of steps for warmup (ε=1)
            anneal_frac: fraction for ε: 1→0.1
            stabilize_frac: fraction for ε=0.1
        """
        self.total_steps = total_steps
        self.epsilon_min = epsilon_min
        self.warmup_end = int(total_steps * warmup_frac)
        self.anneal_end = self.warmup_end + int(total_steps * anneal_frac)
        self.stabilize_end = self.anneal_end + int(total_steps * stabilize_frac)
        # Remaining steps: final anneal from 0.1 → epsilon_min

        # Per-module epsilon overrides
        self._module_schedules: Dict[str, Dict] = {}

    def register_module(self, name: str, category: str = 'tropical'):
        """Register a module with a tropicalization category.

        Categories:
        - 'tropical': follows the main schedule (TropicalAttention, TropicalSSM)
        - 'bridge':   half-speed anneal (SheafDiffusion, GistExtractor)
        - 'classical': stays at ε=1 (CliffordFFN, LayerNorm, Embeddings)
        """
        self._module_schedules[name] = {'category': category}

    def get_epsilon(self, step: int, module_name: Optional[str] = None) -> float:
        """Get the current ε for a given step and optional module."""
        category = 'tropical'
        if module_name and module_name in self._module_schedules:
            category = self._module_schedules[module_name]['category']

        if category == 'classical':
            return 1.0

        base_eps = self._base_epsilon(step)

        if category == 'bridge':
            # Bridge modules anneal at half speed
            return 0.5 * (1.0 + base_eps)

        return base_eps

    def _base_epsilon(self, step: int) -> float:
        """Compute base epsilon for the tropical schedule."""
        if step < self.warmup_end:
            return 1.0
        elif step < self.anneal_end:
            # Cosine anneal from 1.0 to 0.1
            progress = (step - self.warmup_end) / max(
                self.anneal_end - self.warmup_end, 1)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        elif step < self.stabilize_end:
            return 0.1
        else:
            # Final anneal from 0.1 to epsilon_min
            progress = (step - self.stabilize_end) / max(
                self.total_steps - self.stabilize_end, 1)
            progress = min(progress, 1.0)
            return self.epsilon_min + (0.1 - self.epsilon_min) * 0.5 * (
                1 + math.cos(math.pi * progress))


def tropical_logsumexp(x: Tensor, dim: int, epsilon: float = 1.0) -> Tensor:
    """Maslov-parameterized logsumexp.

    At ε=1: standard logsumexp
    At ε→0: max (tropical addition)

    f_ε(x) = ε · logsumexp(x / ε)
    """
    if epsilon >= 0.99:
        return torch.logsumexp(x, dim=dim)
    elif epsilon < 0.01:
        return x.max(dim=dim).values
    else:
        return epsilon * torch.logsumexp(x / epsilon, dim=dim)


def tropical_softmax(x: Tensor, dim: int, epsilon: float = 1.0) -> Tensor:
    """Maslov-parameterized softmax.

    At ε=1: standard softmax
    At ε→0: hardmax (one-hot argmax)
    """
    if epsilon >= 0.99:
        return torch.softmax(x, dim=dim)
    elif epsilon < 0.01:
        idx = x.argmax(dim=dim, keepdim=True)
        return torch.zeros_like(x).scatter_(dim, idx, 1.0)
    else:
        return torch.softmax(x / epsilon, dim=dim)


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 6: Tropical RG Flow Optimizer
# ═══════════════════════════════════════════════════════════════════════════

class TropicalRGFlowOptimizer:
    """Multi-scale optimization via RG coarse-graining in parameter space.

    Inspired by the multigrid method in numerical PDEs, this optimizer
    operates at multiple scales:

    1. Coarsen: Pool parameter groups into coarse representations
    2. Optimize coarse: Find optimal direction in coarse parameter space
    3. Refine: Lift the coarse update back to fine parameter space
    4. Polish: Fine-tune fine parameters

    The RG flow in parameter space:
        θ^{(l+1)} = Pool(θ^{(l)})           (coarsen)
        θ*^{(l+1)} = argmin L_{l+1}(θ)      (optimize coarse)
        θ^{(l)} += Upsample(θ*^{(l+1)})     (refine)

    This connects to the model's built-in RG pooling: the optimizer's
    coarse-graining mirrors the architecture's spatial coarse-graining.
    """

    def __init__(self, model: nn.Module, fine_optimizer: Optimizer,
                 coarse_lr: float = 1e-2, n_coarse_steps: int = 3,
                 coarsen_ratio: float = 0.5):
        """
        Args:
            model: the TSRN model
            fine_optimizer: base optimizer for fine-scale updates
            coarse_lr: learning rate for coarse-scale optimization
            n_coarse_steps: gradient steps at the coarse scale
            coarsen_ratio: fraction of parameters to keep in coarse rep
        """
        self.model = model
        self.fine_optimizer = fine_optimizer
        self.coarse_lr = coarse_lr
        self.n_coarse_steps = n_coarse_steps
        self.coarsen_ratio = coarsen_ratio

        # Build parameter groups for multi-scale
        self._param_scales = self._classify_parameter_scales()

    def _classify_parameter_scales(self) -> Dict[str, str]:
        """Classify parameters into scale levels based on their module.

        Scale 1 (fine): attention Q/K/V, sheaf diffusion, reservoir
        Scale 2 (coarse): s2_blocks parameters
        Global: embeddings, layer norms, head
        """
        scales = {}
        for name, _ in self.model.named_parameters():
            if 's1_block' in name:
                scales[name] = 'fine'
            elif 's2_block' in name:
                scales[name] = 'coarse'
            elif 'rg_pool' in name:
                scales[name] = 'bridge'
            else:
                scales[name] = 'global'
        return scales

    def _coarsen_gradients(self) -> Dict[str, Tensor]:
        """Pool fine-scale gradients into coarse representations.

        Uses max-pooling (tropical!) over gradient dimensions to extract
        the dominant optimization direction at the coarse scale.
        """
        coarse_grads = {}
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            scale = self._param_scales.get(name, 'global')
            if scale == 'fine' and p.grad.dim() >= 2:
                # Max-pool gradients along the first dimension
                # This extracts the dominant gradient direction (tropical!)
                k = max(1, int(p.grad.shape[0] * self.coarsen_ratio))
                # Take top-k gradient rows by magnitude
                norms = p.grad.view(p.grad.shape[0], -1).norm(dim=1)
                _, top_idx = norms.topk(k)
                coarse_grads[name] = (p.grad[top_idx], top_idx)
            else:
                coarse_grads[name] = (p.grad.clone(), None)
        return coarse_grads

    def _refine_and_apply(self, coarse_grads: Dict[str, Tuple]):
        """Lift coarse gradients back to fine scale and apply update."""
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name not in coarse_grads or p.grad is None:
                    continue
                grad, indices = coarse_grads[name]
                if indices is not None:
                    # Scatter coarse gradient back to fine positions
                    update = torch.zeros_like(p.grad)
                    update[indices] = grad
                    p.add_(update, alpha=-self.coarse_lr)
                else:
                    p.add_(grad, alpha=-self.coarse_lr)

    def step(self, loss_fn: Callable, x: Tensor, y: Tensor):
        """One RG flow optimization step.

        1. Compute gradients at fine scale
        2. Coarsen gradients
        3. Apply coarse updates
        4. Fine-tune with base optimizer
        """
        # Fine-scale gradient computation
        self.fine_optimizer.zero_grad(set_to_none=True)
        _, loss = loss_fn(x, y)
        loss.backward()

        # Coarsen
        coarse_grads = self._coarsen_gradients()

        # Coarse optimization steps
        for _ in range(self.n_coarse_steps):
            self._refine_and_apply(coarse_grads)

        # Fine-tune with base optimizer
        self.fine_optimizer.zero_grad(set_to_none=True)
        _, loss_fine = loss_fn(x, y)
        loss_fine.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.fine_optimizer.step()

        return loss.item(), loss_fine.item()


# ═══════════════════════════════════════════════════════════════════════════
#  Unified Tropical Hybrid Trainer
# ═══════════════════════════════════════════════════════════════════════════

class TropicalHybridTrainer:
    """Unified trainer that combines all 6 tropical optimization phases.

    Usage:
        trainer = TropicalHybridTrainer(model, dataset, device, config)
        for step in range(n_steps):
            metrics = trainer.train_step(step)
    """

    def __init__(self, model: nn.Module, dataset, device: torch.device,
                 total_steps: int = 100000, batch_size: int = 8,
                 lr_max: float = 2e-4, lr_min_ratio: float = 0.1,
                 warmup_steps: int = 4000, weight_decay: float = 0.1,
                 grad_accum_steps: int = 1,
                 use_tropical_subgradient: bool = True,
                 use_mirror_descent: bool = False,
                 use_path_optimizer: bool = False,
                 use_geodesic_lr: bool = True,
                 use_maslov_schedule: bool = True,
                 use_rg_flow: bool = False,
                 path_explore_every: int = 50,
                 rg_flow_every: int = 20):
        """
        Configure which tropical optimization phases to use.
        Not all phases should run simultaneously — recommended combos:

        Conservative:  subgradient + geodesic_lr + maslov
        Moderate:      subgradient + geodesic_lr + maslov + rg_flow
        Aggressive:    all phases (experimental)
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps

        # Classify parameters into tropical vs classical
        tropical_params, classical_params = self._classify_params(model)

        # Phase 1: Tropical Subgradient Optimizer
        self.use_tropical_subgradient = use_tropical_subgradient
        if use_tropical_subgradient:
            param_groups = [
                {'params': list(tropical_params.values()),
                 'tropical': True, 'lr': lr_max,
                 'weight_decay': weight_decay},
                {'params': list(classical_params.values()),
                 'tropical': False, 'lr': lr_max,
                 'weight_decay': weight_decay},
            ]
            self.optimizer = TropicalSubgradientOptimizer(
                param_groups, lr=lr_max, weight_decay=weight_decay)
        elif use_mirror_descent:
            param_groups = [
                {'params': list(tropical_params.values()),
                 'tropical': True, 'lr': lr_max,
                 'tropical_lr': lr_max * 0.5,
                 'weight_decay': weight_decay},
                {'params': list(classical_params.values()),
                 'tropical': False, 'lr': lr_max,
                 'weight_decay': weight_decay},
            ]
            self.optimizer = TropicalMirrorDescent(
                param_groups, lr=lr_max, weight_decay=weight_decay)
        else:
            all_params = {**tropical_params, **classical_params}
            decay_params = [p for n, p in all_params.items() if p.dim() >= 2]
            no_decay_params = [p for n, p in all_params.items() if p.dim() < 2]
            self.optimizer = torch.optim.AdamW([
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ], lr=lr_max, betas=(0.9, 0.95))

        # Phase 3: Path optimizer (runs periodically)
        self.use_path_optimizer = use_path_optimizer
        self.path_explore_every = path_explore_every
        if use_path_optimizer:
            self.path_optimizer = TropicalPathOptimizer(
                model, self.optimizer, n_neighbors=3)

        # Phase 4: Tropical Geodesic LR
        self.use_geodesic_lr = use_geodesic_lr
        if use_geodesic_lr:
            self.lr_scheduler = TropicalGeodesicLR(
                self.optimizer, model,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                lr_min_ratio=lr_min_ratio)
        else:
            self.lr_scheduler = None

        # Phase 5: Maslov Dequantization Schedule
        self.use_maslov_schedule = use_maslov_schedule
        if use_maslov_schedule:
            self.maslov_schedule = MaslovDequantizationSchedule(
                total_steps=total_steps)
            self._register_maslov_modules(model)
        else:
            self.maslov_schedule = None

        # Phase 6: RG Flow Optimizer (runs periodically)
        self.use_rg_flow = use_rg_flow
        self.rg_flow_every = rg_flow_every
        if use_rg_flow:
            self.rg_flow = TropicalRGFlowOptimizer(
                model, self.optimizer, coarse_lr=lr_max * 0.1)

    @staticmethod
    def _classify_params(model: nn.Module) -> Tuple[Dict, Dict]:
        """Classify parameters as tropical or classical."""
        tropical = {}
        classical = {}
        tropical_modules = {
            'attn', 'tropical_ssm', 'rg_pool', 'rope', 'alibi',
        }
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_tropical = any(tm in name.lower() for tm in tropical_modules)
            if is_tropical:
                tropical[name] = p
            else:
                classical[name] = p
        return tropical, classical

    def _register_maslov_modules(self, model: nn.Module):
        """Register modules with the Maslov schedule."""
        for name, module in model.named_modules():
            cls_name = type(module).__name__
            if cls_name in ('TropicalAttention', 'TropicalSSM'):
                self.maslov_schedule.register_module(name, 'tropical')
            elif cls_name in ('SheafDiffusion', 'SheafRotorDiffusion',
                              'GistExtractor'):
                self.maslov_schedule.register_module(name, 'bridge')
            else:
                self.maslov_schedule.register_module(name, 'classical')

    def train_step(self, step: int) -> Dict[str, float]:
        """Execute one training step with all active tropical phases.

        Returns dict of metrics.
        """
        self.model.train()
        metrics = {}

        # Get batch
        x, y = self.dataset.batch("train", self.batch_size, self.device)

        # Phase 5: Get current Maslov epsilon
        if self.maslov_schedule:
            epsilon = self.maslov_schedule.get_epsilon(step)
            metrics['maslov_epsilon'] = epsilon
            # Store epsilon on model for use in forward pass
            if hasattr(self.model, '_maslov_epsilon'):
                self.model._maslov_epsilon = epsilon

        # Phase 3: Periodic path exploration
        if (self.use_path_optimizer and step > 0 and
                step % self.path_explore_every == 0):
            loss_val, neighbor_loss = self.path_optimizer.step(
                self.model, x, y)
            metrics['path_loss'] = loss_val
            metrics['path_neighbor_loss'] = neighbor_loss
            path_stats = self.path_optimizer.get_path_stats()
            metrics.update({f'path_{k}': v for k, v in path_stats.items()})
            return metrics

        # Phase 6: Periodic RG flow optimization
        if (self.use_rg_flow and step > 0 and
                step % self.rg_flow_every == 0):
            coarse_loss, fine_loss = self.rg_flow.step(
                self.model, x, y)
            metrics['rg_coarse_loss'] = coarse_loss
            metrics['rg_fine_loss'] = fine_loss
            return metrics

        # Standard training step with gradient accumulation
        self.optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(self.grad_accum_steps):
            x, y = self.dataset.batch("train", self.batch_size, self.device)
            _, loss = self.model(x, y)
            (loss / self.grad_accum_steps).backward()
            accum_loss += loss.item() / self.grad_accum_steps

        gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Phase 4: Update geodesic LR before step
        if self.lr_scheduler and self.use_geodesic_lr:
            self.lr_scheduler.step_with_tropical_distance()
            metrics['lr'] = self.optimizer.param_groups[0]['lr']

        self.optimizer.step()

        metrics['train_loss'] = accum_loss
        metrics['grad_norm'] = float(gnorm)

        # Phase 1: Log tropical sparsity
        if self.use_tropical_subgradient and hasattr(self.optimizer,
                                                      'get_tropical_sparsity'):
            sparsity = self.optimizer.get_tropical_sparsity()
            for k, v in sparsity.items():
                metrics[f'tropical_sparsity_{k}'] = v.get('mean_sparsity', 0)

        return metrics

    def get_current_epsilon(self, step: int,
                            module_name: str = None) -> float:
        """Get current Maslov epsilon for a module."""
        if self.maslov_schedule:
            return self.maslov_schedule.get_epsilon(step, module_name)
        return 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience: drop-in replacement for train_gist / train_model
# ═══════════════════════════════════════════════════════════════════════════

def train_with_tropical_optimizer(
    model: nn.Module,
    dataset,
    device: torch.device,
    n_steps: int = 100000,
    batch_size: int = 8,
    lr_max: float = 2e-4,
    warmup_steps: int = 4000,
    weight_decay: float = 0.1,
    eval_every: int = 2000,
    grad_accum_steps: int = 1,
    use_phases: str = "conservative",
    eval_fn=None,
) -> List[Dict]:
    """Drop-in training function using tropical optimizers.

    Args:
        use_phases: "conservative" | "moderate" | "aggressive"
            conservative: subgradient + geodesic_lr + maslov
            moderate: + rg_flow
            aggressive: + path_optimizer + mirror_descent
    """
    phase_config = {
        'conservative': dict(
            use_tropical_subgradient=True,
            use_mirror_descent=False,
            use_path_optimizer=False,
            use_geodesic_lr=True,
            use_maslov_schedule=True,
            use_rg_flow=False,
        ),
        'moderate': dict(
            use_tropical_subgradient=True,
            use_mirror_descent=False,
            use_path_optimizer=False,
            use_geodesic_lr=True,
            use_maslov_schedule=True,
            use_rg_flow=True,
        ),
        'aggressive': dict(
            use_tropical_subgradient=False,
            use_mirror_descent=True,
            use_path_optimizer=True,
            use_geodesic_lr=True,
            use_maslov_schedule=True,
            use_rg_flow=True,
        ),
    }

    config = phase_config.get(use_phases, phase_config['conservative'])

    trainer = TropicalHybridTrainer(
        model=model, dataset=dataset, device=device,
        total_steps=n_steps, batch_size=batch_size,
        lr_max=lr_max, warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        grad_accum_steps=grad_accum_steps,
        **config,
    )

    log = []
    import time
    t0 = time.time()

    for step in range(1, n_steps + 1):
        metrics = trainer.train_step(step)

        if step % eval_every == 0 or step == 1:
            if eval_fn:
                eval_metrics = eval_fn(model, dataset, device)
                metrics.update(eval_metrics)

            elapsed = time.time() - t0
            metrics['step'] = step
            metrics['time_s'] = round(elapsed, 1)
            log.append(metrics)

            loss = metrics.get('train_loss', 0)
            bpc = loss / math.log(2)
            val_bpc = metrics.get('val_bpc', 0)
            eps = metrics.get('maslov_epsilon', 1.0)
            lr = metrics.get('lr', lr_max)
            print(f"  Step {step:>6}  loss={loss:.4f}  bpc={bpc:.4f}  "
                  f"val_bpc={val_bpc:.4f}  ε={eps:.3f}  lr={lr:.6f}")

    return log
