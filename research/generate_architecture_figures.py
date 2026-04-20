"""
Generate publication-quality figures for the TSRN Architecture Deep Dive.

Each function produces one or more figures saved to research/figures/.
Run: python research/generate_architecture_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Color palette
C_TROP = '#E63946'      # tropical red
C_CLASS = '#457B9D'      # classical blue
C_SHEAF = '#2A9D8F'      # sheaf teal
C_CLIFF = '#E9C46A'      # clifford gold
C_PADIC = '#264653'      # p-adic dark
C_RG = '#F4A261'         # RG orange
C_BG = '#F8F9FA'         # light background
C_GRID = '#DEE2E6'       # grid color

plt.rcParams.update({
    'figure.facecolor': C_BG,
    'axes.facecolor': '#FFFFFF',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': C_GRID,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


# ═══════════════════════════════════════════════════════════════════
# Figure 1: Tropical vs Classical Inner Product
# ═══════════════════════════════════════════════════════════════════

def fig_tropical_inner_product():
    """Compare tropical logsumexp inner product with standard dot product."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: 1D comparison — tropical vs classical as a function of one coordinate
    x = np.linspace(-3, 3, 300)
    y_vals = [0.5, 1.0, 2.0]

    ax = axes[0]
    for y in y_vals:
        # Standard dot product: q*k = x*y (linear)
        dot = x * y
        # Tropical: logsumexp(q_i + k_i) for 2D case ~ max(x+y, 0)
        trop = np.log(np.exp(x + y) + np.exp(0))
        ax.plot(x, dot, '--', color=C_CLASS, alpha=0.5, linewidth=1.5)
        ax.plot(x, trop, '-', color=C_TROP, alpha=0.8, linewidth=2,
                label=f'tropical (y={y})' if y == 1.0 else '')
    ax.plot(x, x * 1.0, '--', color=C_CLASS, linewidth=1.5, label='classical (y=1)')
    ax.set_xlabel('q coordinate')
    ax.set_ylabel('score')
    ax.set_title('A. Tropical vs Classical Score')
    ax.legend(fontsize=9)

    # Panel B: 2D tropical variety — decision boundaries
    ax = axes[1]
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))

    # Tropical polynomial: max(x+1, y+0.5, -x-y+2)
    z1 = xx + 1
    z2 = yy + 0.5
    z3 = -xx - yy + 2
    z = np.maximum(np.maximum(z1, z2), z3)

    # Which region "wins"
    region = np.zeros_like(xx, dtype=int)
    region[z2 > z1] = 1
    region[z3 > np.maximum(z1, z2)] = 2

    cmap = LinearSegmentedColormap.from_list('trop',
        [(0, '#FADBD8'), (0.5, '#D5F5E3'), (1.0, '#D6EAF8')])
    ax.contourf(xx, yy, region.astype(float), levels=[-0.5, 0.5, 1.5, 2.5],
                colors=['#FADBD8', '#D5F5E3', '#D6EAF8'], alpha=0.7)
    # Decision boundaries (tropical variety)
    ax.contour(xx, yy, z1 - z2, levels=[0], colors=[C_TROP], linewidths=2.5)
    ax.contour(xx, yy, z1 - z3, levels=[0], colors=[C_TROP], linewidths=2.5)
    ax.contour(xx, yy, z2 - z3, levels=[0], colors=[C_TROP], linewidths=2.5)
    ax.set_xlabel('$q_1 + k_1$')
    ax.set_ylabel('$q_2 + k_2$')
    ax.set_title('B. Tropical Variety (Decision Boundaries)')
    ax.annotate('Region 1\n$q_1+k_1$ wins', xy=(2, -2), fontsize=9,
                ha='center', color=C_TROP, fontweight='bold')
    ax.annotate('Region 2\n$q_2+k_2$ wins', xy=(-2, 2), fontsize=9,
                ha='center', color=C_SHEAF, fontweight='bold')
    ax.annotate('Region 3\nmixed', xy=(-1, -1), fontsize=9,
                ha='center', color=C_CLASS, fontweight='bold')

    # Panel C: Sparsity pattern — top-k masking
    ax = axes[2]
    np.random.seed(42)
    T = 16
    scores = np.random.randn(T, T)
    # Apply causal mask
    mask = np.triu(np.ones((T, T)), k=1)
    scores[mask == 1] = -10
    # Top-k (k=4) per row
    sparse_scores = np.full_like(scores, -10)
    for i in range(T):
        row = scores[i]
        valid = np.where(row > -10)[0]
        if len(valid) > 0:
            topk = valid[np.argsort(row[valid])[-min(4, len(valid)):]]
            sparse_scores[i, topk] = row[topk]

    im = ax.imshow(sparse_scores, cmap='RdYlBu_r', aspect='auto',
                   vmin=-3, vmax=3)
    ax.set_xlabel('Key position')
    ax.set_ylabel('Query position')
    ax.set_title('C. Tropical Top-k Sparse Attention')
    plt.colorbar(im, ax=ax, shrink=0.8, label='score')

    fig.suptitle('Tropical Sparse Attention — Geometry & Sparsity',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'tropical_attention.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [1/12] tropical_attention.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 2: RoPE — Rotary Position Embeddings
# ═══════════════════════════════════════════════════════════════════

def fig_rope():
    """Visualize RoPE rotations in the complex plane."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Rotation of a single vector at different positions
    ax = axes[0]
    theta = 1.0 / (10000 ** (0 / 64))  # frequency for dim pair 0
    q = np.array([1.0, 0.3])  # original query vector

    positions = [0, 4, 8, 16, 32]
    colors_pos = plt.cm.viridis(np.linspace(0.2, 0.9, len(positions)))

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    circle = plt.Circle((0, 0), np.sqrt(q[0]**2 + q[1]**2),
                         fill=False, color=C_GRID, linewidth=1, linestyle='--')
    ax.add_patch(circle)

    for pos, col in zip(positions, colors_pos):
        angle = pos * theta
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        q_rot = np.array([q[0]*cos_a - q[1]*sin_a,
                          q[0]*sin_a + q[1]*cos_a])
        ax.annotate('', xy=q_rot, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=col, lw=2))
        ax.plot(*q_rot, 'o', color=col, markersize=6)
        ax.annotate(f't={pos}', xy=q_rot, fontsize=8,
                    xytext=(5, 5), textcoords='offset points', color=col)

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_title('A. RoPE: Query Rotation by Position')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')

    # Panel B: Frequency spectrum across dimensions
    ax = axes[1]
    d_model = 64
    dims = np.arange(0, d_model // 2)
    freqs = 1.0 / (10000 ** (2 * dims / d_model))
    wavelengths = 2 * np.pi / freqs

    ax.semilogy(dims * 2, wavelengths, 'o-', color=C_TROP, markersize=3)
    ax.set_xlabel('Dimension pair index')
    ax.set_ylabel('Wavelength (positions)')
    ax.set_title('B. RoPE Frequency Spectrum')
    ax.axhline(y=2*np.pi, color=C_GRID, linestyle='--', alpha=0.5)
    ax.annotate('wavelength = 2π', xy=(d_model//4, 2*np.pi*1.1),
                fontsize=9, color='gray')

    # Panel C: Relative position encoding — dot product depends on t_q - t_k
    ax = axes[2]
    T = 32
    # Compute attention pattern from RoPE alone
    rope_scores = np.zeros((T, T))
    q_base = np.random.randn(d_model)
    k_base = np.random.randn(d_model)

    for i in range(T):
        for j in range(T):
            score = 0.0
            for dim_pair in range(d_model // 2):
                freq = 1.0 / (10000 ** (2 * dim_pair / d_model))
                angle_q = i * freq
                angle_k = j * freq
                # Rotated dot product
                qi = q_base[2*dim_pair] * np.cos(angle_q) - q_base[2*dim_pair+1] * np.sin(angle_q)
                qi2 = q_base[2*dim_pair] * np.sin(angle_q) + q_base[2*dim_pair+1] * np.cos(angle_q)
                ki = k_base[2*dim_pair] * np.cos(angle_k) - k_base[2*dim_pair+1] * np.sin(angle_k)
                ki2 = k_base[2*dim_pair] * np.sin(angle_k) + k_base[2*dim_pair+1] * np.cos(angle_k)
                score += qi * ki + qi2 * ki2
            rope_scores[i, j] = score

    im = ax.imshow(rope_scores, cmap='coolwarm', aspect='auto')
    ax.set_xlabel('Key position')
    ax.set_ylabel('Query position')
    ax.set_title('C. RoPE Score Pattern (relative position)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Rotary Position Embeddings (RoPE)',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'rope.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [2/12] rope.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 3: ALiBi — Attention with Linear Biases
# ═══════════════════════════════════════════════════════════════════

def fig_alibi():
    """Visualize ALiBi bias patterns across heads."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    T = 32
    H = 8

    # Panel A: Bias slopes per head
    ax = axes[0]
    slopes = [2 ** (-8.0 / H * (h + 1)) for h in range(H)]
    colors_h = plt.cm.plasma(np.linspace(0.15, 0.85, H))
    distances = np.arange(T)
    for h, (slope, col) in enumerate(zip(slopes, colors_h)):
        bias = -slope * distances
        ax.plot(distances, bias, '-', color=col, linewidth=2,
                label=f'head {h} (m={slope:.4f})')
    ax.set_xlabel('Distance |i - j|')
    ax.set_ylabel('Bias (added to score)')
    ax.set_title('A. ALiBi Bias per Head')
    ax.legend(fontsize=7, ncol=2, loc='lower left')

    # Panel B: Full bias matrix for head 0 (steepest) and head 7 (shallowest)
    ax = axes[1]
    bias_mat_h0 = np.zeros((T, T))
    bias_mat_h7 = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            bias_mat_h0[i, j] = -slopes[0] * abs(i - j)
            bias_mat_h7[i, j] = -slopes[-1] * abs(i - j)
    # Show head 0
    im = ax.imshow(bias_mat_h0, cmap='RdPu_r', aspect='auto')
    ax.set_xlabel('Key position')
    ax.set_ylabel('Query position')
    ax.set_title(f'B. Head 0 Bias (steep, m={slopes[0]:.4f})')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel C: Head 7 (shallowest)
    ax = axes[2]
    im = ax.imshow(bias_mat_h7, cmap='RdPu_r', aspect='auto')
    ax.set_xlabel('Key position')
    ax.set_ylabel('Query position')
    ax.set_title(f'C. Head 7 Bias (gentle, m={slopes[-1]:.4f})')
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('ALiBi — Multi-Head Distance Bias',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'alibi.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [3/12] alibi.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 4: Sheaf Diffusion — Fiber Bundle
# ═══════════════════════════════════════════════════════════════════

def fig_sheaf_diffusion():
    """Visualize sheaf diffusion as a fiber bundle with restriction maps."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Panel A: Fiber bundle visualization
    ax = axes[0]
    T = 6
    d = 3  # show 3D stalks
    np.random.seed(7)

    # Base space (horizontal)
    base_y = 0
    ax.plot([0, T-1], [base_y, base_y], '-', color='gray', linewidth=3, alpha=0.5)

    # Stalks (vertical fibers)
    stalk_heights = np.random.randn(T, d) * 0.3 + np.array([0.5, 0.8, 1.1])
    colors_d = [C_TROP, C_SHEAF, C_CLIFF]

    for t in range(T):
        ax.plot([t, t], [base_y, 1.5], '-', color=C_GRID, linewidth=1, alpha=0.5)
        for dim in range(d):
            ax.plot(t, stalk_heights[t, dim], 'o', color=colors_d[dim],
                    markersize=8, zorder=5)

    # Restriction maps (arrows between adjacent stalks)
    for t in range(T - 1):
        for dim in range(d):
            ax.annotate('', xy=(t+1, stalk_heights[t+1, dim]),
                        xytext=(t, stalk_heights[t, dim]),
                        arrowprops=dict(arrowstyle='->', color=colors_d[dim],
                                        alpha=0.5, linewidth=1.5,
                                        connectionstyle='arc3,rad=0.15'))

    ax.set_xlim(-0.5, T-0.5)
    ax.set_ylim(-0.3, 1.7)
    ax.set_xlabel('Sequence position (base space)')
    ax.set_ylabel('Feature dimensions (fiber)')
    ax.set_title('A. Fiber Bundle: Stalks & Restriction Maps')
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=c, markersize=8, label=f'dim {i}')
                       for i, c in enumerate(colors_d)]
    ax.legend(handles=legend_elements, fontsize=9)

    # Panel B: Sheaf Laplacian energy landscape
    ax = axes[1]
    # Simulate sheaf energy: ||R_d x_i - x_{i+d}||^2
    T_sim = 50
    np.random.seed(42)
    x = np.cumsum(np.random.randn(T_sim, 2) * 0.1, axis=0)
    # Energy at each position
    energy = np.zeros(T_sim)
    for t in range(1, T_sim):
        energy[t] = np.sum((x[t] - x[t-1])**2)

    # Before diffusion
    ax.plot(range(T_sim), energy, '-', color=C_TROP, linewidth=2, alpha=0.7,
            label='Before diffusion')

    # After diffusion (smoothed)
    energy_smooth = np.convolve(energy, np.ones(5)/5, mode='same')
    energy_smooth *= 0.5  # diffusion reduces energy
    ax.plot(range(T_sim), energy_smooth, '-', color=C_SHEAF, linewidth=2,
            label='After diffusion')

    ax.fill_between(range(T_sim), energy, energy_smooth, alpha=0.15, color=C_SHEAF)
    ax.set_xlabel('Sequence position')
    ax.set_ylabel('Local inconsistency energy')
    ax.set_title('B. Sheaf Laplacian Energy Reduction')
    ax.legend(fontsize=9)

    # Panel C: Causal offset pattern
    ax = axes[2]
    window = 3
    T_show = 10
    # Causal offsets: [-3, -2, -1, 0]
    offsets = list(range(-window, 1))

    # Draw connectivity matrix
    conn = np.zeros((T_show, T_show))
    for t in range(T_show):
        for d in offsets:
            nb = t + d
            if 0 <= nb < T_show:
                conn[t, nb] = 1.0 / (abs(d) + 1)

    im = ax.imshow(conn, cmap='Greens', aspect='auto')
    ax.set_xlabel('Source position (neighbor)')
    ax.set_ylabel('Target position')
    ax.set_title(f'C. Causal Sheaf Connectivity (w={window})')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Connection strength')
    # Mark the diagonal and causal region
    for t in range(T_show):
        for d in offsets:
            nb = t + d
            if 0 <= nb < T_show:
                ax.plot(nb, t, 's', color='white', markersize=3, alpha=0.5)

    fig.suptitle('Sheaf Diffusion — Local Coherence via Restriction Maps',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'sheaf_diffusion.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [4/12] sheaf_diffusion.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 5: Clifford FFN — Geometric Product + SwiGLU
# ═══════════════════════════════════════════════════════════════════

def fig_clifford_ffn():
    """Visualize the Clifford geometric product and SwiGLU activation."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Geometric product in the complex plane
    ax = axes[0]
    theta_range = np.linspace(0, 2*np.pi, 200)
    r_fixed = 1.0

    # Input: z = r*e^{iθ}
    # Geometric product: z² = r² * e^{2iθ}
    # grade-0 = r²*cos(2θ), grade-2 = r²*sin(2θ)

    x_in = r_fixed * np.cos(theta_range)
    y_in = r_fixed * np.sin(theta_range)
    x_out = r_fixed**2 * np.cos(2*theta_range)
    y_out = r_fixed**2 * np.sin(2*theta_range)

    ax.plot(x_in, y_in, '-', color=C_CLASS, linewidth=2, alpha=0.5, label='Input z')
    ax.plot(x_out, y_out, '-', color=C_CLIFF, linewidth=2.5, label='Output z²')

    # Mark specific points
    for angle, label in [(0, '0°'), (np.pi/4, '45°'), (np.pi/2, '90°'), (np.pi, '180°')]:
        xi, yi = np.cos(angle), np.sin(angle)
        xo, yo = np.cos(2*angle), np.sin(2*angle)
        ax.plot(xi, yi, 'o', color=C_CLASS, markersize=8)
        ax.plot(xo, yo, 's', color=C_CLIFF, markersize=8)
        ax.annotate('', xy=(xo, yo), xytext=(xi, yi),
                    arrowprops=dict(arrowstyle='->', color='gray',
                                    alpha=0.4, connectionstyle='arc3,rad=0.3'))

    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Real (grade-0)')
    ax.set_ylabel('Imaginary (grade-2)')
    ax.set_title('A. Clifford Geometric Product (z → z²)')
    ax.legend(fontsize=9)

    # Panel B: Grade decomposition surface
    ax = axes[1]
    r_range = np.linspace(0, 2, 100)
    i_range = np.linspace(0, 2, 100)
    R, I = np.meshgrid(r_range, i_range)
    grade0 = R**2 - I**2
    grade2 = 2 * R * I

    c0 = ax.contourf(R, I, grade0, levels=20, cmap='RdBu_r', alpha=0.8)
    ax.contour(R, I, grade0, levels=[0], colors=['black'], linewidths=2)
    ax.set_xlabel('Real component (r)')
    ax.set_ylabel('Imaginary component (i)')
    ax.set_title('B. Grade-0: r² − i² (scalar similarity)')
    plt.colorbar(c0, ax=ax, shrink=0.8)
    ax.annotate('r²=i²\n(zero crossing)', xy=(1.0, 1.0), fontsize=9,
                ha='center', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # Panel C: SwiGLU vs Sigmoid comparison
    ax = axes[2]
    x = np.linspace(-4, 4, 300)

    sigmoid = 1 / (1 + np.exp(-x))
    silu = x * sigmoid  # SiLU = x * σ(x)
    relu = np.maximum(0, x)
    gelu = x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    ax.plot(x, sigmoid, '--', color=C_CLASS, linewidth=2, label='Sigmoid', alpha=0.7)
    ax.plot(x, relu, ':', color='gray', linewidth=1.5, label='ReLU', alpha=0.5)
    ax.plot(x, gelu, '-.', color=C_SHEAF, linewidth=1.5, label='GELU', alpha=0.7)
    ax.plot(x, silu, '-', color=C_CLIFF, linewidth=2.5, label='SiLU (SwiGLU gate)')

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_title('C. Activation Functions: SiLU vs Others')
    ax.legend(fontsize=9)
    ax.set_ylim(-1.5, 4)

    fig.suptitle('Clifford Geometric FFN with SwiGLU Gating',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'clifford_ffn.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [5/12] clifford_ffn.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 6: RG Pool — MERA Coarse-Graining
# ═══════════════════════════════════════════════════════════════════

def fig_rg_pool():
    """Visualize MERA tensor network and causal coarse-graining."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: MERA-style tensor network
    ax = axes[0]
    ax.set_xlim(-1, 9)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')

    # Fine scale (bottom) — 8 tokens
    fine_x = np.arange(8)
    fine_y = np.zeros(8)
    for x in fine_x:
        ax.plot(x, 0, 'o', color=C_CLASS, markersize=14, zorder=5)
        ax.text(x, -0.35, f'$x_{{{x}}}$', ha='center', fontsize=9)

    # Disentangle layer (crosses between pairs)
    dis_y = 1.2
    for j in range(4):
        x_left = 2*j
        x_right = 2*j + 1
        mid_x = (x_left + x_right) / 2

        # Draw X (disentangle)
        ax.plot([x_left, x_right], [0.3, dis_y - 0.3], '-', color=C_RG, linewidth=1.5)
        ax.plot([x_right, x_left], [0.3, dis_y - 0.3], '-', color=C_RG, linewidth=1.5)

        # Disentangle node
        ax.plot(mid_x, dis_y * 0.5, 'D', color=C_RG, markersize=10, zorder=5)

    # Pool layer — merge pairs into coarse tokens
    pool_y = 2.5
    for j in range(4):
        x_left = 2*j
        x_right = 2*j + 1
        mid_x = (x_left + x_right) / 2

        # Lines from disentangle to pool
        ax.plot([x_left, mid_x], [dis_y, pool_y], '-', color=C_SHEAF, linewidth=2)
        ax.plot([x_right, mid_x], [dis_y, pool_y], '-', color=C_SHEAF, linewidth=2)

        # Coarse tokens
        ax.plot(mid_x, pool_y, 's', color=C_TROP, markersize=14, zorder=5)
        ax.text(mid_x, pool_y + 0.35, f'$c_{{{j}}}$', ha='center', fontsize=10,
                color=C_TROP, fontweight='bold')

    # Causal arrows showing info flow
    ax.annotate('Causal:\n$c_j$ uses $x_{{2j-1}}, x_{{2j}}$',
                xy=(6.5, 3.5), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

    ax.set_title('A. MERA Tensor Network (Disentangle + Pool)', fontsize=13)
    ax.axis('off')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_CLASS,
               markersize=10, label='Fine tokens'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=C_RG,
               markersize=10, label='Disentangle'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_TROP,
               markersize=10, label='Coarse tokens'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # Panel B: Causal pairing diagram — show the shift
    ax = axes[1]
    ax.set_xlim(-1.5, 9)
    ax.set_ylim(-1, 6)
    ax.set_aspect('equal')

    # Original sequence
    y_orig = 5
    for t in range(8):
        ax.plot(t, y_orig, 'o', color=C_CLASS, markersize=12, zorder=5)
        ax.text(t, y_orig + 0.35, f'$x_{{{t}}}$', ha='center', fontsize=9)
    ax.text(-1.2, y_orig, 'Original:', ha='right', fontsize=10, fontweight='bold')

    # Shifted sequence (prepend 0)
    y_shift = 3.5
    tokens_shifted = ['0'] + [f'$x_{{{t}}}$' for t in range(8)]
    colors_shifted = ['lightgray'] + [C_CLASS]*8
    for t in range(9):
        ax.plot(t - 0.5, y_shift, 'o', color=colors_shifted[t], markersize=12, zorder=5)
        ax.text(t - 0.5, y_shift + 0.35, tokens_shifted[t], ha='center', fontsize=8)
    ax.text(-1.7, y_shift, 'Shifted:', ha='right', fontsize=10, fontweight='bold')

    # Pairs
    y_pair = 1.5
    pair_labels = [
        ('0', '$x_0$'), ('$x_1$', '$x_2$'), ('$x_3$', '$x_4$'), ('$x_5$', '$x_6$')
    ]
    for j, (left, right) in enumerate(pair_labels):
        x_pos = j * 2 + 0.5
        # Draw bracket
        ax.plot([x_pos-0.4, x_pos-0.4, x_pos+0.4, x_pos+0.4],
                [y_pair+0.5, y_pair-0.2, y_pair-0.2, y_pair+0.5],
                '-', color=C_TROP, linewidth=2)
        ax.text(x_pos, y_pair - 0.5, f'$c_{{{j}}}$', ha='center',
                fontsize=11, color=C_TROP, fontweight='bold')
        ax.text(x_pos, y_pair + 0.15, f'({left}, {right})', ha='center', fontsize=8)
        # max index annotation
        ax.text(x_pos, y_pair - 0.9, f'max idx={2*j}', ha='center',
                fontsize=7, color='gray')

        # Arrows from shifted to pair
        ax.annotate('', xy=(x_pos - 0.2, y_pair + 0.6),
                    xytext=(j*2 - 0.5, y_shift - 0.3),
                    arrowprops=dict(arrowstyle='->', color=C_RG, alpha=0.5))
        ax.annotate('', xy=(x_pos + 0.2, y_pair + 0.6),
                    xytext=(j*2 + 0.5, y_shift - 0.3),
                    arrowprops=dict(arrowstyle='->', color=C_RG, alpha=0.5))

    ax.set_title('B. Causal Pairing: Prepend Zero + Stride', fontsize=13)
    ax.axis('off')

    fig.suptitle('RG Coarse-Graining — MERA-Inspired Multi-Scale',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'rg_pool.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [6/12] rg_pool.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 7: p-adic Tree & Ultrametric Distance
# ═══════════════════════════════════════════════════════════════════

def fig_padic_tree():
    """Visualize the p-adic binary tree and ultrametric distance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Binary tree visualization
    ax = axes[0]
    ax.set_xlim(-1, 17)
    ax.set_ylim(-0.5, 5)

    # Level 0 (root)
    root_x, root_y = 8, 4.5
    ax.plot(root_x, root_y, 'o', color=C_PADIC, markersize=16, zorder=5)
    ax.text(root_x, root_y + 0.3, 'Root', ha='center', fontsize=9, fontweight='bold')

    # Level 1
    l1 = [(4, 3.5), (12, 3.5)]
    for x, y in l1:
        ax.plot(x, y, 'o', color=C_PADIC, markersize=13, zorder=5, alpha=0.8)
    ax.plot([root_x, l1[0][0]], [root_y, l1[0][1]], '-', color=C_PADIC, linewidth=2)
    ax.plot([root_x, l1[1][0]], [root_y, l1[1][1]], '-', color=C_PADIC, linewidth=2)
    ax.text(6, 4.1, '0', fontsize=10, color=C_PADIC)
    ax.text(10, 4.1, '1', fontsize=10, color=C_PADIC)

    # Level 2
    l2 = [(2, 2.5), (6, 2.5), (10, 2.5), (14, 2.5)]
    for x, y in l2:
        ax.plot(x, y, 'o', color=C_PADIC, markersize=11, zorder=5, alpha=0.7)
    for parent, children in [((4, 3.5), [(2, 2.5), (6, 2.5)]),
                              ((12, 3.5), [(10, 2.5), (14, 2.5)])]:
        for child in children:
            ax.plot([parent[0], child[0]], [parent[1], child[1]],
                    '-', color=C_PADIC, linewidth=1.5, alpha=0.7)

    # Level 3 (leaves)
    l3 = [(1, 1.5), (3, 1.5), (5, 1.5), (7, 1.5),
           (9, 1.5), (11, 1.5), (13, 1.5), (15, 1.5)]
    leaf_colors = plt.cm.Set2(np.linspace(0, 1, 8))
    for idx, (x, y) in enumerate(l3):
        ax.plot(x, y, 'o', color=leaf_colors[idx], markersize=14, zorder=5,
                markeredgecolor=C_PADIC, markeredgewidth=1.5)
        ax.text(x, y - 0.4, f'slot {idx}', ha='center', fontsize=7)

    for parent, children in [((2, 2.5), [(1, 1.5), (3, 1.5)]),
                              ((6, 2.5), [(5, 1.5), (7, 1.5)]),
                              ((10, 2.5), [(9, 1.5), (11, 1.5)]),
                              ((14, 2.5), [(13, 1.5), (15, 1.5)])]:
        for child in children:
            ax.plot([parent[0], child[0]], [parent[1], child[1]],
                    '-', color=C_PADIC, linewidth=1, alpha=0.5)

    # Distance annotations
    ax.annotate('d(0,1) = 2⁻²\n(close)',
                xy=(2, 0.8), fontsize=9, ha='center', color=C_SHEAF,
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
    ax.annotate('d(0,4) = 2⁻¹\n(far)',
                xy=(5, 0.3), fontsize=9, ha='center', color=C_RG,
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
    ax.annotate('d(0,7) = 2⁰\n(very far)',
                xy=(11, 0.3), fontsize=9, ha='center', color=C_TROP,
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

    ax.set_title('A. p-adic Binary Tree (depth=3, M=8 slots)', fontsize=13)
    ax.axis('off')

    # Panel B: Ultrametric distance matrix
    ax = axes[1]
    M = 8
    dist = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i == j:
                dist[i, j] = 0
            else:
                # p-adic distance: 2^(-shared_prefix_length)
                xor = i ^ j
                prefix = 0
                for bit in range(2, -1, -1):
                    if (i >> bit) & 1 == (j >> bit) & 1:
                        prefix += 1
                    else:
                        break
                dist[i, j] = 2 ** (-prefix)

    im = ax.imshow(dist, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel('Memory slot')
    ax.set_ylabel('Memory slot')
    ax.set_title('B. Ultrametric Distance Matrix')
    plt.colorbar(im, ax=ax, shrink=0.8, label='p-adic distance')
    for i in range(M):
        for j in range(M):
            ax.text(j, i, f'{dist[i,j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if dist[i,j] > 0.6 else 'black')

    fig.suptitle('p-adic Hierarchical Memory — Ultrametric Structure',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'padic_tree.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [7/12] padic_tree.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 8: Echo State Reservoir — Spectral Dynamics + GRU Gating
# ═══════════════════════════════════════════════════════════════════

def fig_reservoir():
    """Visualize reservoir dynamics, spectral radius, and GRU gating."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Eigenvalue spectrum of reservoir weight matrix
    ax = axes[0]
    np.random.seed(42)
    d = 50
    W = np.random.randn(d, d) * 0.1
    W[np.random.rand(d, d) > 0.1] = 0  # 90% sparse
    # Scale to spectral radius ~0.95
    eigs = np.linalg.eigvals(W)
    sr = np.max(np.abs(eigs))
    W = W * (0.95 / sr)
    eigs = np.linalg.eigvals(W)

    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), '--', color=C_GRID, linewidth=1,
            label='Unit circle')
    ax.plot(0.95*np.cos(theta), 0.95*np.sin(theta), '-', color=C_TROP,
            linewidth=1.5, alpha=0.5, label='ρ = 0.95')
    ax.scatter(eigs.real, eigs.imag, c=np.abs(eigs), cmap='plasma',
               s=30, zorder=5, edgecolors='black', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('A. Reservoir Eigenvalue Spectrum')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)

    # Panel B: GRU gate dynamics over time
    ax = axes[1]
    T = 50
    np.random.seed(7)
    x_input = np.zeros(T)
    x_input[10:15] = 1.0  # input burst
    x_input[30:33] = 0.7

    # Simulate simple GRU-like gating
    h = np.zeros(T)
    z_gate = np.zeros(T)
    for t in range(1, T):
        z = 1 / (1 + np.exp(-(0.5 * x_input[t] - 0.3 * h[t-1])))
        z_gate[t] = z
        candidate = np.tanh(0.8 * h[t-1] + x_input[t])
        h[t] = (1 - z) * h[t-1] + z * candidate

    ax.fill_between(range(T), x_input, alpha=0.2, color=C_CLASS, label='Input')
    ax.plot(range(T), h, '-', color=C_TROP, linewidth=2, label='Hidden state h')
    ax.plot(range(T), z_gate, '--', color=C_SHEAF, linewidth=1.5, label='Update gate z')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Value')
    ax.set_title('B. GRU-Gated Reservoir: State & Gate Dynamics')
    ax.legend(fontsize=9)

    # Panel C: Spectral radius vs behavior regime
    ax = axes[2]
    rho_range = np.linspace(0.1, 1.5, 100)
    # Memory capacity ~ 1/|1-ρ| (simplified)
    memory = 1.0 / np.abs(1 - rho_range + 0.05)
    # Stability ~ max(0, 1-ρ)
    stability = np.maximum(0, 1 - rho_range)

    ax.plot(rho_range, memory / memory.max(), '-', color=C_TROP, linewidth=2.5,
            label='Memory capacity')
    ax.plot(rho_range, stability, '-', color=C_CLASS, linewidth=2.5,
            label='Stability')
    ax.fill_between([0.85, 1.0], 0, 1, alpha=0.15, color=C_CLIFF,
                    label='Edge of chaos (sweet spot)')
    ax.axvline(0.95, color=C_RG, linewidth=2, linestyle='--',
               label='Target ρ = 0.95')
    ax.set_xlabel('Spectral radius ρ')
    ax.set_ylabel('Normalized measure')
    ax.set_title('C. Spectral Radius Trade-off')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(0.1, 1.5)
    ax.set_ylim(0, 1.1)

    fig.suptitle('GRU-Gated Echo State Reservoir',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'reservoir.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [8/12] reservoir.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 9: Tropical SSM — Max-Plus Recurrence
# ═══════════════════════════════════════════════════════════════════

def fig_tropical_ssm():
    """Visualize max-plus recurrence dynamics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    T = 30
    d_state = 4
    np.random.seed(42)

    # Panel A: State trajectories under max-plus recurrence
    ax = axes[0]
    A_diag = np.array([-0.05, -0.1, -0.15, -0.2])  # decay rates
    x_input = np.random.randn(T, d_state) * 0.5

    h = np.zeros((T, d_state))
    which_won = np.zeros((T, d_state), dtype=int)  # 0=state, 1=input

    for t in range(1, T):
        state_path = A_diag + h[t-1]
        input_path = x_input[t]
        h[t] = np.maximum(state_path, input_path)
        which_won[t] = (input_path > state_path).astype(int)

    colors_state = [C_TROP, C_CLASS, C_SHEAF, C_CLIFF]
    for dim in range(d_state):
        ax.plot(range(T), h[:, dim], '-', color=colors_state[dim],
                linewidth=2, label=f'h[{dim}]', alpha=0.8)
        # Mark where input won
        input_won = np.where(which_won[:, dim] == 1)[0]
        ax.plot(input_won, h[input_won, dim], 'v', color=colors_state[dim],
                markersize=5, alpha=0.6)

    ax.set_xlabel('Time step')
    ax.set_ylabel('State value')
    ax.set_title('A. Max-Plus State Trajectories')
    ax.legend(fontsize=8, ncol=2)

    # Panel B: "Which path won" visualization
    ax = axes[1]
    im = ax.imshow(which_won.T, cmap=LinearSegmentedColormap.from_list(
        'win', [(0, '#D5F5E3'), (1, '#FADBD8')]),
        aspect='auto', interpolation='nearest')
    ax.set_xlabel('Time step')
    ax.set_ylabel('State dimension')
    ax.set_title('B. Winning Path: State (green) vs Input (red)')
    ax.set_yticks(range(d_state))
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#D5F5E3', label='State path (A + h_{t-1})'),
        mpatches.Patch(facecolor='#FADBD8', label='Input path (B·x_t)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

    # Panel C: Comparison of standard vs tropical SSM
    ax = axes[2]
    # Standard linear: h_t = 0.95*h_{t-1} + x_t
    # Tropical: h_t = max(-0.05 + h_{t-1}, x_t)
    h_linear = np.zeros(T)
    h_trop = np.zeros(T)
    x_1d = np.random.randn(T) * 0.3
    x_1d[10] = 2.0  # spike

    for t in range(1, T):
        h_linear[t] = 0.95 * h_linear[t-1] + x_1d[t]
        h_trop[t] = max(-0.05 + h_trop[t-1], x_1d[t])

    ax.plot(range(T), h_linear, '-', color=C_CLASS, linewidth=2.5,
            label='Standard SSM (linear)')
    ax.plot(range(T), h_trop, '-', color=C_TROP, linewidth=2.5,
            label='Tropical SSM (max-plus)')
    ax.fill_between(range(T), x_1d, alpha=0.15, color='gray', label='Input')
    ax.axvline(10, color='gray', linestyle=':', alpha=0.5)
    ax.annotate('Spike input', xy=(10, 2.0), fontsize=9,
                xytext=(15, 2.5), arrowprops=dict(arrowstyle='->', color='gray'))
    ax.set_xlabel('Time step')
    ax.set_ylabel('State value')
    ax.set_title('C. Standard vs Tropical SSM Response')
    ax.legend(fontsize=9)

    fig.suptitle('Tropical SSM — Max-Plus Linear Recurrence',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'tropical_ssm.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [9/12] tropical_ssm.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 10: Maslov Dequantization — Classical ↔ Tropical
# ═══════════════════════════════════════════════════════════════════

def fig_maslov_dequantization():
    """Visualize the Maslov dequantization schedule and softmax→hardmax transition."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: softmax at different temperatures (epsilon)
    ax = axes[0]
    x = np.linspace(-3, 3, 7)
    epsilons = [1.0, 0.5, 0.2, 0.1, 0.01]
    colors_eps = plt.cm.magma(np.linspace(0.2, 0.9, len(epsilons)))

    for eps, col in zip(epsilons, colors_eps):
        if eps < 0.02:
            # hardmax
            probs = np.zeros_like(x)
            probs[np.argmax(x)] = 1.0
        else:
            scaled = x / eps
            scaled -= scaled.max()
            probs = np.exp(scaled) / np.exp(scaled).sum()
        ax.bar(np.arange(len(x)) + (epsilons.index(eps) - 2) * 0.15,
               probs, width=0.14, color=col, alpha=0.8,
               label=f'ε={eps}')

    ax.set_xlabel('Input index')
    ax.set_ylabel('Probability')
    ax.set_title('A. Softmax → Hardmax as ε → 0')
    ax.legend(fontsize=8)
    ax.set_xticks(range(len(x)))

    # Panel B: ε schedule over training
    ax = axes[1]
    total_steps = 100000
    steps = np.arange(total_steps)

    warmup_end = int(total_steps * 0.1)
    anneal_end = warmup_end + int(total_steps * 0.4)
    stabilize_end = anneal_end + int(total_steps * 0.3)

    eps_schedule = np.ones(total_steps)
    # Warmup: ε = 1
    # Anneal: 1 → 0.1
    anneal_range = np.arange(warmup_end, anneal_end)
    progress = (anneal_range - warmup_end) / (anneal_end - warmup_end)
    eps_schedule[warmup_end:anneal_end] = 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))
    # Stabilize: 0.1
    eps_schedule[anneal_end:stabilize_end] = 0.1
    # Final: 0.1 → 0.01
    final_range = np.arange(stabilize_end, total_steps)
    progress_f = (final_range - stabilize_end) / max(total_steps - stabilize_end, 1)
    eps_schedule[stabilize_end:] = 0.01 + (0.1 - 0.01) * 0.5 * (1 + np.cos(np.pi * progress_f))

    ax.plot(steps / 1000, eps_schedule, '-', color=C_TROP, linewidth=2.5)
    ax.axhline(1.0, color=C_CLASS, linestyle='--', alpha=0.3, label='Classical (ε=1)')
    ax.axhline(0.01, color=C_TROP, linestyle='--', alpha=0.3, label='Tropical (ε→0)')

    # Phase annotations
    ax.axvspan(0, warmup_end/1000, alpha=0.08, color=C_CLASS)
    ax.axvspan(warmup_end/1000, anneal_end/1000, alpha=0.08, color=C_RG)
    ax.axvspan(anneal_end/1000, stabilize_end/1000, alpha=0.08, color=C_SHEAF)
    ax.axvspan(stabilize_end/1000, total_steps/1000, alpha=0.08, color=C_TROP)

    ax.text(warmup_end/2000, 0.85, 'Warmup', ha='center', fontsize=9, fontweight='bold')
    ax.text((warmup_end+anneal_end)/2000, 0.85, 'Anneal', ha='center', fontsize=9, fontweight='bold')
    ax.text((anneal_end+stabilize_end)/2000, 0.85, 'Stabilize', ha='center', fontsize=9, fontweight='bold')
    ax.text((stabilize_end+total_steps)/2000, 0.85, 'Final', ha='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('Training step (×1000)')
    ax.set_ylabel('Maslov ε')
    ax.set_title('B. Dequantization Schedule')
    ax.legend(fontsize=9)

    # Panel C: logsumexp interpolation
    ax = axes[2]
    x_range = np.linspace(-3, 3, 300)
    # f_ε(x, 0) = ε * log(exp(x/ε) + exp(0/ε))
    for eps, col in zip([1.0, 0.5, 0.2, 0.05], colors_eps[:4]):
        if eps < 0.02:
            y = np.maximum(x_range, 0)
        else:
            y = eps * np.log(np.exp(x_range / eps) + np.exp(0 / eps))
        ax.plot(x_range, y, '-', color=col, linewidth=2, label=f'ε={eps}')

    ax.plot(x_range, np.maximum(x_range, 0), '--', color='black', linewidth=1,
            alpha=0.5, label='max(x, 0) [pure tropical]')
    ax.set_xlabel('x')
    ax.set_ylabel('f_ε(x, 0)')
    ax.set_title('C. Maslov Interpolation: logsumexp → max')
    ax.legend(fontsize=8)

    fig.suptitle('Maslov Dequantization — Classical ↔ Tropical Bridge',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'maslov_dequantization.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [10/12] maslov_dequantization.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 11: Full Architecture Diagram
# ═══════════════════════════════════════════════════════════════════

def fig_full_architecture():
    """High-level architecture diagram with color-coded components."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')

    def draw_box(ax, x, y, w, h, label, color, sublabel=None, alpha=0.3):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=alpha,
                                        edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.1 if sublabel else 0),
                label, ha='center', va='center', fontsize=11, fontweight='bold')
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.2, sublabel,
                    ha='center', va='center', fontsize=8, color='gray')

    def draw_arrow(ax, x1, y1, x2, y2, color='black'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    linewidth=2, shrinkA=3, shrinkB=3))

    # Title
    ax.text(5, 19.5, 'TSRN Full Architecture', ha='center', fontsize=18,
            fontweight='bold')
    ax.text(5, 19.1, 'Tropical Sheaf Renormalization Network',
            ha='center', fontsize=12, color='gray')

    # Input
    draw_box(ax, 3.5, 18.0, 3, 0.6, 'Input Tokens', '#E8E8E8', 'B × T integers')
    draw_arrow(ax, 5, 18.0, 5, 17.5)

    # Embedding
    draw_box(ax, 3, 16.9, 4, 0.5, 'Embedding + Position', C_PADIC, alpha=0.2)
    draw_arrow(ax, 5, 16.9, 5, 16.4)

    # Scale 1 blocks
    y_s1 = 12.8
    draw_box(ax, 1, y_s1, 8, 3.5, '', C_CLASS, alpha=0.08)
    ax.text(5, y_s1 + 3.3, 'SCALE 1 (Fine: T tokens) × n_blocks',
            ha='center', fontsize=12, fontweight='bold', color=C_CLASS)

    components_s1 = [
        ('TropicalAttention', C_TROP, 'RoPE + ALiBi + causal mask + top-k'),
        ('SheafDiffusion', C_SHEAF, 'causal offsets, restriction maps'),
        ('GRU-Reservoir', C_RG, 'selective memory, edge-of-chaos'),
        ('TropicalSSM', C_TROP, 'max-plus recurrence (block 0 only)'),
        ('CliffordFFN', C_CLIFF, 'geometric product + SwiGLU'),
        ('PAdicMemory', C_PADIC, 'binary tree, 2^d slots'),
    ]
    for i, (name, color, desc) in enumerate(components_s1):
        y_comp = y_s1 + 2.8 - i * 0.48
        draw_box(ax, 1.5, y_comp, 7, 0.42, name, color, desc, alpha=0.25)

    draw_arrow(ax, 5, y_s1, 5, y_s1 - 0.5)

    # RG Pool
    y_rg = y_s1 - 1.2
    draw_box(ax, 2.5, y_rg, 5, 0.6, 'RG Pool', C_RG,
             'T → T/2 (MERA: disentangle + pool)', alpha=0.35)
    draw_arrow(ax, 5, y_rg, 5, y_rg - 0.5)

    # Scale 2
    y_s2 = y_rg - 2.5
    draw_box(ax, 1.5, y_s2, 7, 1.8, '', C_TROP, alpha=0.06)
    ax.text(5, y_s2 + 1.6, 'SCALE 2 (Coarse: T/2 tokens) × n_blocks',
            ha='center', fontsize=12, fontweight='bold', color=C_TROP)

    components_s2 = [
        ('TropicalAttention', C_TROP, 'same architecture, coarser scale'),
        ('SheafDiffusion', C_SHEAF, 'causal'),
        ('CliffordFFN', C_CLIFF, 'SwiGLU'),
        ('PAdicAttention', C_PADIC, 'tree-structured (last block)'),
    ]
    for i, (name, color, desc) in enumerate(components_s2):
        y_comp = y_s2 + 1.3 - i * 0.38
        draw_box(ax, 2, y_comp, 6, 0.32, name, color, desc, alpha=0.25)

    draw_arrow(ax, 5, y_s2, 5, y_s2 - 0.5)

    # Upsample + Fusion
    y_fuse = y_s2 - 1.0
    draw_box(ax, 2, y_fuse, 6, 0.6, 'Upsample + Gated Fusion', C_CLIFF,
             'T/2 → T, gate = σ(W·[x; xc_up])', alpha=0.3)

    # Skip connection from Scale 1 output
    ax.annotate('', xy=(1.5, y_fuse + 0.3), xytext=(0.5, y_s1),
                arrowprops=dict(arrowstyle='->', color=C_CLASS,
                                linewidth=2, linestyle='--',
                                connectionstyle='arc3,rad=-0.15'))
    ax.text(0.3, (y_fuse + y_s1) / 2, 'skip\nconnection',
            fontsize=8, color=C_CLASS, ha='center', rotation=90)

    draw_arrow(ax, 5, y_fuse, 5, y_fuse - 0.5)

    # Output
    y_out = y_fuse - 1.0
    draw_box(ax, 3, y_out, 4, 0.5, 'LayerNorm → Head', '#E8E8E8',
             'Linear → logits (B × T × V)')
    draw_arrow(ax, 5, y_out, 5, y_out - 0.5)

    # Loss
    draw_box(ax, 3.5, y_out - 0.8, 3, 0.5, 'Cross-Entropy Loss', '#F8D7DA')

    # Color legend
    y_leg = 1.0
    legend_items = [
        ('Tropical (max-plus)', C_TROP),
        ('Sheaf (local coherence)', C_SHEAF),
        ('Clifford (geometric)', C_CLIFF),
        ('p-adic (hierarchical)', C_PADIC),
        ('RG (multi-scale)', C_RG),
    ]
    for i, (label, color) in enumerate(legend_items):
        rect = mpatches.FancyBboxPatch((0.5, y_leg - i * 0.35), 0.3, 0.25,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, alpha=0.4)
        ax.add_patch(rect)
        ax.text(1.0, y_leg - i * 0.35 + 0.12, label, fontsize=9, va='center')

    fig.savefig(os.path.join(FIGDIR, 'full_architecture.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [11/12] full_architecture.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 12: Five Geometric Spaces — Relationships
# ═══════════════════════════════════════════════════════════════════

def fig_geometric_spaces():
    """Visualize the 5 mathematical spaces and their connections."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(-2, 12)
    ax.set_ylim(-1, 11)
    ax.axis('off')

    ax.text(5, 10.5, 'Five Mathematical Spaces of TSRN',
            ha='center', fontsize=18, fontweight='bold')

    # Draw 5 spaces as large circles
    spaces = [
        (2, 7.5, C_TROP, 'Tropical\nGeometry', '(ℝ, max, +)\nmax-plus algebra\nPWL functions'),
        (8, 7.5, C_SHEAF, 'Sheaf\nTheory', 'Fiber bundles\nRestriction maps\nLocal coherence'),
        (1, 3.5, C_CLIFF, 'Clifford\nAlgebra', 'Cl(1,0) rotors\nGeometric products\nGrade decomposition'),
        (9, 3.5, C_PADIC, 'p-adic\nAnalysis', 'Ultrametric trees\nHierarchical distance\nNon-Archimedean'),
        (5, 1, C_RG, 'Renormalization\nGroup', 'MERA tensor network\nScale hierarchy\nCoarse-graining'),
    ]

    for x, y, color, name, desc in spaces:
        circle = plt.Circle((x, y), 1.3, facecolor=color, alpha=0.2,
                            edgecolor=color, linewidth=2.5)
        ax.add_patch(circle)
        ax.text(x, y + 0.3, name, ha='center', va='center',
                fontsize=12, fontweight='bold', color=color)
        ax.text(x, y - 0.5, desc, ha='center', va='center',
                fontsize=8, color='gray')

    # Draw connections with labels
    connections = [
        (2, 7.5, 8, 7.5, 'Maslov\ndequantization', 0.0),
        (2, 7.5, 1, 3.5, 'Tropical\nscores →\nClifford FFN', -0.2),
        (2, 7.5, 5, 1, 'Tropical\nSSM feeds\nRG pool', 0.0),
        (8, 7.5, 1, 3.5, 'Sheaf coherence\n→ rotor transform', 0.0),
        (8, 7.5, 9, 3.5, 'Local coherence\n+ hierarchical\nmemory', 0.2),
        (1, 3.5, 5, 1, 'Clifford gist\nrotation →\nscale bridge', 0.0),
        (9, 3.5, 5, 1, 'p-adic memory\n→ multi-scale\nretrieval', 0.0),
        (1, 3.5, 9, 3.5, 'Geometric\nfeatures →\ntree encoding', 0.0),
    ]

    for x1, y1, x2, y2, label, rad in connections:
        # Shorten arrows to not overlap circles
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        x1s = x1 + dx / dist * 1.35
        y1s = y1 + dy / dist * 1.35
        x2s = x2 - dx / dist * 1.35
        y2s = y2 - dy / dist * 1.35

        ax.annotate('', xy=(x2s, y2s), xytext=(x1s, y1s),
                    arrowprops=dict(arrowstyle='<->', color='gray',
                                    linewidth=1.5, alpha=0.6,
                                    connectionstyle=f'arc3,rad={rad}'))
        # Label at midpoint
        mx = (x1 + x2) / 2 + rad * 1.5
        my = (y1 + y2) / 2 + rad * 0.5
        ax.text(mx, my, label, ha='center', va='center', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8),
                color='gray')

    # Central bridging concept
    ax.text(5, 5.5, 'Maslov ε\nBridge', ha='center', va='center',
            fontsize=14, fontweight='bold', color=C_TROP,
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                      edgecolor=C_TROP, linewidth=2))
    ax.annotate('', xy=(5, 6.3), xytext=(5, 5.9),
                arrowprops=dict(arrowstyle='->', color=C_TROP, linewidth=2))
    ax.annotate('', xy=(5, 4.7), xytext=(5, 5.1),
                arrowprops=dict(arrowstyle='->', color=C_TROP, linewidth=2))

    ax.text(5, 6.5, 'ε=1: Classical\n(softmax, smooth)', ha='center',
            fontsize=9, color=C_CLASS)
    ax.text(5, 4.3, 'ε→0: Tropical\n(hardmax, PWL)', ha='center',
            fontsize=9, color=C_TROP)

    fig.savefig(os.path.join(FIGDIR, 'geometric_spaces.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [12/12] geometric_spaces.png")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\nGenerating TSRN Architecture Figures...")
    print(f"Output directory: {FIGDIR}\n")

    fig_tropical_inner_product()
    fig_rope()
    fig_alibi()
    fig_sheaf_diffusion()
    fig_clifford_ffn()
    fig_rg_pool()
    fig_padic_tree()
    fig_reservoir()
    fig_tropical_ssm()
    fig_maslov_dequantization()
    fig_full_architecture()
    fig_geometric_spaces()

    print(f"\nDone! {12} figures saved to {FIGDIR}/")


if __name__ == "__main__":
    main()
