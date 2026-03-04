#!/usr/bin/env python3
"""
Experiment 4: LoRA Weight and SVD Analysis

For both stage 0 and stage 1 models:
  - Layer-wise importance: Frobenius norm of BA for each layer/module (heatmap)
  - Cross-stage comparison: same norms at both checkpoints
  - Singular value spectrum: 16 singular values of BA per module
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

STAGE0_ADAPTER = PROJECT_ROOT / "mistralai_Mistral_Small_3.2_24B_Instruct_2506_20260302_175851" / "final" / "adapter_model.safetensors"
STAGE1_ADAPTER = PROJECT_ROOT / "mistralai_Mistral_Small_3.2_24B_Instruct_2506_20260302_223714" / "final" / "adapter_model.safetensors"

MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MODULE_DISPLAY = {
    "q_proj": "Q", "k_proj": "K", "v_proj": "V", "o_proj": "O",
    "gate_proj": "Gate", "up_proj": "Up", "down_proj": "Down"
}
MODULE_GROUP = {
    "q_proj": "attn", "k_proj": "attn", "v_proj": "attn", "o_proj": "attn",
    "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"
}


def nonzero_layer_summary(norm_grid: np.ndarray) -> str:
    """Return a compact summary of layers whose row norm is non-zero."""
    row_sums = norm_grid.sum(axis=1)
    nz = np.where(row_sums > 0)[0]
    if len(nz) == 0:
        return "non-zero layers: none"
    return f"non-zero layers: {int(nz.min())}-{int(nz.max())} ({len(nz)}/{norm_grid.shape[0]})"


def load_lora_weights(path: Path):
    """
    Load LoRA A and B matrices from a safetensors file.
    Returns dict: {(layer_idx, module_name): (A_matrix, B_matrix)}
    """
    print(f"Loading adapter weights from: {path}")
    weights = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        # Identify all (layer, module) pairs
        for key in keys:
            # Key format: base_model.model.model.layers.{i}.self_attn.{proj}.lora_A.weight
            #          or base_model.model.model.layers.{i}.mlp.{proj}.lora_A.weight
            # Mistral3 adapters can also include vision_tower LoRA tensors; this
            # experiment targets language-model blocks only.
            if ".vision_tower." in key:
                continue
            if "lora_A" not in key:
                continue
            parts = key.split(".")
            # find 'layers' position
            try:
                layers_idx = parts.index("layers")
            except ValueError:
                continue
            layer_num = int(parts[layers_idx + 1])
            # module name is 2 positions after the sublayer (self_attn or mlp)
            module_name = parts[-3]  # e.g. q_proj
            if module_name not in MODULES:
                continue
            a_key = key
            b_key = key.replace("lora_A", "lora_B")
            if b_key not in keys:
                continue
            A = f.get_tensor(a_key).float().numpy()
            B = f.get_tensor(b_key).float().numpy()
            weights[(layer_num, module_name)] = (A, B)

    print(f"  Loaded {len(weights)} (layer, module) pairs")
    return weights


def frobenius_norm_efficient(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute ||BA||_F efficiently WITHOUT forming the full d×d matrix ΔW.

    ΔW = B @ A  where B ∈ R^{d×r}, A ∈ R^{r×d}, r=16 (LoRA rank).

    ||BA||_F^2 = trace(A^T B^T B A) = trace(K @ M)
    where K = B^T B (r×r) and M = A @ A^T (r×r).
    This is O(r²d) and avoids the d×d intermediate matrix.
    """
    r = A.shape[0]
    K = B.T @ B   # r×r
    M = A @ A.T   # r×r
    return float(np.sqrt(max(0.0, np.trace(K @ M))))


def compute_singular_values_efficient(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute the r=16 singular values of ΔW = BA efficiently.

    Uses QR decomposition of B to reduce to an r×d_in SVD:
      B = Q @ R  (Q orthonormal, R upper-triangular, both r×r or d×r/r×r)
      BA = Q (RA)  → singular values of BA = singular values of RA (r×d_in)
    SVD of a 16×d matrix is fast (O(r²d)) and memory-efficient.
    """
    Q, R = np.linalg.qr(B, mode="reduced")  # Q: d×r, R: r×r
    RA = R @ A  # r×d_in (16×d_in)
    _, s, _ = np.linalg.svd(RA, full_matrices=False)  # s: r values
    return s


def analyse_adapter(weights: dict, label: str):
    """Compute Frobenius norms and SVDs for all (layer, module) pairs."""
    layer_ids = sorted(set(k[0] for k in weights))
    num_layers = max(layer_ids) + 1

    # Frobenius norm grid: rows=layers, cols=modules
    norm_grid = np.zeros((num_layers, len(MODULES)))
    svd_data = {}  # (layer, module) -> singular values array

    for (layer_num, module_name), (A, B) in weights.items():
        fn = frobenius_norm_efficient(A, B)
        col = MODULES.index(module_name)
        norm_grid[layer_num, col] = fn

        sv = compute_singular_values_efficient(A, B)
        svd_data[(layer_num, module_name)] = sv

    print(f"[{label}] Norm grid shape: {norm_grid.shape}")
    return norm_grid, svd_data


def plot_heatmap(norm_grid: np.ndarray, title: str, save_path: Path):
    """Plot layer × module Frobenius norm heatmap."""
    fig, ax = plt.subplots(figsize=(10, max(6, norm_grid.shape[0] // 3)))
    im = ax.imshow(norm_grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(MODULES)))
    ax.set_xticklabels([MODULE_DISPLAY[m] for m in MODULES], fontsize=10)
    ax.set_ylabel("Layer index")
    ax.set_xlabel("Module")
    ax.set_title(title)
    ax.text(
        0.01, 1.01, nonzero_layer_summary(norm_grid),
        transform=ax.transAxes, fontsize=9, ha="left", va="bottom"
    )
    plt.colorbar(im, ax=ax, label="‖ΔW‖_F")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"  Saved heatmap: {save_path}")


def plot_comparison_heatmap(norm0: np.ndarray, norm1: np.ndarray, save_path: Path):
    """Side-by-side heatmaps for stage0 vs stage1."""
    n_layers = max(norm0.shape[0], norm1.shape[0])
    # Pad if needed
    if norm0.shape[0] < n_layers:
        norm0 = np.pad(norm0, ((0, n_layers - norm0.shape[0]), (0, 0)))
    if norm1.shape[0] < n_layers:
        norm1 = np.pad(norm1, ((0, n_layers - norm1.shape[0]), (0, 0)))

    vmax = max(norm0.max(), norm1.max())
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, n_layers // 3)))

    for ax, data, label in [(axes[0], norm0, "Stage 1"), (axes[1], norm1, "Stage 2")]:
        im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(MODULES)))
        ax.set_xticklabels([MODULE_DISPLAY[m] for m in MODULES], fontsize=9)
        ax.set_ylabel("Layer index")
        ax.set_title(f"{label} ‖ΔW‖_F")
        ax.text(
            0.01, 1.01, nonzero_layer_summary(data),
            transform=ax.transAxes, fontsize=8, ha="left", va="bottom"
        )
        plt.colorbar(im, ax=ax)

    # Difference plot
    diff = norm1 - norm0
    absmax = np.abs(diff).max() or 1e-9
    im3 = axes[2].imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-absmax, vmax=absmax)
    axes[2].set_xticks(range(len(MODULES)))
    axes[2].set_xticklabels([MODULE_DISPLAY[m] for m in MODULES], fontsize=9)
    axes[2].set_title("Stage2 − Stage1")
    plt.colorbar(im3, ax=axes[2], label="Δ‖ΔW‖_F")

    plt.suptitle("Cross-Stage LoRA Frobenius Norm Comparison (Stage 1 vs Stage 2)", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"  Saved comparison heatmap: {save_path}")


def plot_singular_value_spectra(svd0: dict, svd1: dict, save_path: Path):
    """
    For each module type, plot the distribution of singular values
    aggregated across all layers (median ± IQR), comparing stage0 vs stage1.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for idx, module in enumerate(MODULES):
        ax = axes[idx]
        for svd_data, label, color in [(svd0, "Stage 1", "#2196F3"), (svd1, "Stage 2", "#F44336")]:
            all_svs = [svd_data[(l, module)] for (l, m) in svd_data if m == module]
            if not all_svs:
                continue
            # Align lengths
            min_len = min(len(sv) for sv in all_svs)
            arr = np.array([sv[:min_len] for sv in all_svs])  # (n_layers, rank)
            median = np.median(arr, axis=0)
            q25 = np.percentile(arr, 25, axis=0)
            q75 = np.percentile(arr, 75, axis=0)
            x = np.arange(1, len(median) + 1)
            ax.plot(x, median, label=label, color=color)
            ax.fill_between(x, q25, q75, alpha=0.2, color=color)

        ax.set_title(f"{MODULE_DISPLAY[module]} ({MODULE_GROUP[module]})")
        ax.set_xlabel("Singular value rank")
        ax.set_ylabel("σ")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    axes[-1].set_visible(False)
    plt.suptitle("Singular Value Spectrum of ΔW = BA per Module (median ± IQR across layers)", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"  Saved singular value spectra: {save_path}")


def plot_per_layer_spectra_for_module(svd0: dict, svd1: dict, module: str, save_path: Path):
    """Plot singular value spectra per layer for a specific module."""
    layers0 = sorted(l for (l, m) in svd0 if m == module)
    layers1 = sorted(l for (l, m) in svd1 if m == module)
    all_layers = sorted(set(layers0) | set(layers1))
    n = len(all_layers)
    if n == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, svd_data, stage_label, layers in [
        (axes[0], svd0, "Stage 1", layers0),
        (axes[1], svd1, "Stage 2", layers1)
    ]:
        cmap = plt.cm.plasma
        for i, layer in enumerate(layers):
            sv = svd_data.get((layer, module))
            if sv is None:
                continue
            color = cmap(i / max(len(layers) - 1, 1))
            ax.plot(np.arange(1, len(sv) + 1), sv, alpha=0.5, color=color, linewidth=0.8)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=len(layers) - 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Layer index")
        ax.set_title(f"{stage_label} — {MODULE_DISPLAY[module]}")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Singular value")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Per-layer singular value spectrum: {module}")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=120)
    plt.close()


def main():
    results = {}

    # Load weights
    weights0 = load_lora_weights(STAGE0_ADAPTER)
    weights1 = load_lora_weights(STAGE1_ADAPTER)

    # Analyse
    norm0, svd0 = analyse_adapter(weights0, "Stage 0")
    norm1, svd1 = analyse_adapter(weights1, "Stage 1")

    # --- Heatmaps ---
    plot_heatmap(norm0, "Stage 1 — Layer × Module Frobenius Norm ‖ΔW‖_F",
                 OUT_DIR / "stage0_frobenius_heatmap.png")
    plot_heatmap(norm1, "Stage 2 — Layer × Module Frobenius Norm ‖ΔW‖_F",
                 OUT_DIR / "stage1_frobenius_heatmap.png")
    plot_comparison_heatmap(norm0, norm1, OUT_DIR / "cross_stage_comparison_heatmap.png")

    # --- Singular value spectra ---
    plot_singular_value_spectra(svd0, svd1, OUT_DIR / "singular_value_spectra.png")

    # Per-layer spectra for q_proj and gate_proj (attention vs MLP representatives)
    for mod in ["q_proj", "gate_proj"]:
        plot_per_layer_spectra_for_module(svd0, svd1, mod,
                                          OUT_DIR / f"per_layer_svd_{mod}.png")

    # --- Numerical summary ---
    # Sorted norms for all layer/module pairs (per stage)
    for stage_label, norm_grid in [("stage0", norm0), ("stage1", norm1)]:
        flat = [(norm_grid[l, c], l, MODULES[c])
                for l in range(norm_grid.shape[0])
                for c in range(norm_grid.shape[1])]
        flat.sort(reverse=True)
        results[f"{stage_label}_all_norms"] = [
            {"norm": float(n), "layer": l, "module": m}
            for n, l, m in flat
        ]
        # Keep top-10 for quick inspection and backward compatibility.
        results[f"{stage_label}_top10_norms"] = results[f"{stage_label}_all_norms"][:10]

    # Average norm per module type (attn vs mlp)
    for stage_label, norm_grid in [("stage0", norm0), ("stage1", norm1)]:
        attn_cols = [i for i, m in enumerate(MODULES) if MODULE_GROUP[m] == "attn"]
        mlp_cols = [i for i, m in enumerate(MODULES) if MODULE_GROUP[m] == "mlp"]
        results[f"{stage_label}_avg_norm_attn"] = float(norm_grid[:, attn_cols].mean())
        results[f"{stage_label}_avg_norm_mlp"] = float(norm_grid[:, mlp_cols].mean())
        results[f"{stage_label}_total_norm"] = float(norm_grid.sum())

    # Singular value concentration: ratio of top-2 to sum of all 16
    for stage_label, svd_data in [("stage0", svd0), ("stage1", svd1)]:
        concentration_by_module = {}
        for module in MODULES:
            svs = [svd_data[(l, module)] for (l, m) in svd_data if m == module]
            if not svs:
                continue
            min_len = min(len(sv) for sv in svs)
            arr = np.array([sv[:min_len] for sv in svs])
            mean_sv = arr.mean(axis=0)
            total = mean_sv.sum()
            top2 = mean_sv[:2].sum()
            concentration_by_module[module] = {
                "top2_ratio": float(top2 / total) if total > 0 else 0.0,
                "effective_rank": float((mean_sv.sum() ** 2) / (mean_sv ** 2).sum())
                if (mean_sv ** 2).sum() > 0 else 0.0,
                "mean_sv": mean_sv.tolist()
            }
        results[f"{stage_label}_sv_concentration"] = concentration_by_module

    # Save JSON results
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR / 'results.json'}")

    # Print summary
    print("\n=== SUMMARY ===")
    for stage in ["stage0", "stage1"]:
        print(f"\n[{stage.upper()}]")
        print(f"  Total Frobenius norm: {results[f'{stage}_total_norm']:.4f}")
        print(f"  Avg norm (attention): {results[f'{stage}_avg_norm_attn']:.4f}")
        print(f"  Avg norm (MLP):       {results[f'{stage}_avg_norm_mlp']:.4f}")
        print(f"  Top-3 (layer, module):")
        for entry in results[f"{stage}_all_norms"][:3]:
            print(f"    Layer {entry['layer']:3d} {entry['module']:12s}: {entry['norm']:.4f}")
        print(f"  SV concentration (top-2 ratio):")
        for mod, stats in results[f"{stage}_sv_concentration"].items():
            print(f"    {mod:12s}: top-2/total = {stats['top2_ratio']:.3f}  eff_rank = {stats['effective_rank']:.2f}")

    # Write text summary
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write("=== LoRA Weight SVD Analysis ===\n\n")
        for stage in ["stage0", "stage1"]:
            f.write(f"[{stage.upper()}]\n")
            f.write(f"  Total Frobenius norm sum: {results[f'{stage}_total_norm']:.4f}\n")
            f.write(f"  Avg norm — attention modules: {results[f'{stage}_avg_norm_attn']:.4f}\n")
            f.write(f"  Avg norm — MLP modules:       {results[f'{stage}_avg_norm_mlp']:.4f}\n")
            f.write(f"  All (layer, module, norm), sorted descending:\n")
            for entry in results[f"{stage}_all_norms"]:
                f.write(f"    Layer {entry['layer']:3d}  {entry['module']:12s}  {entry['norm']:.4f}\n")
            f.write(f"  Singular value concentration:\n")
            for mod, stats in results[f"{stage}_sv_concentration"].items():
                f.write(f"    {mod:12s}  top-2/total={stats['top2_ratio']:.3f}  eff_rank={stats['effective_rank']:.2f}\n")
            f.write("\n")

    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
