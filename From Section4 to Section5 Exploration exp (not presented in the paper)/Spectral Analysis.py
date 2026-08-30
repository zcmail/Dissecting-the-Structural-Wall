"""
LoRA Adapter spectral analysis
对全部 32 层的 LoRA 权重做 SVD，计算谱指标，对比有害层 vs 重要层
"""

import os
import gc
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ============================================================================
# 1. 谱分析核心计算
# ============================================================================

def compute_layer_spectrum(
    model,
    num_layers: int = 32,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    对每层每个 LoRA 模块计算 ΔW = B@A·scaling 的 SVD。

    Returns:
        {
            layer_idx: {
                "module_name": {
                    "singular_values": np.ndarray,  # (r,) 奇异值降序排列
                    "U": np.ndarray,                # (out, r) 左奇异向量
                    "V": np.ndarray,                # (in, r) 右奇异向量
                    "delta_W": np.ndarray,          # (out, in) 有效权重矩阵
                }
            }
        }
    """
    spectra = {}

    for layer_idx in range(num_layers):
        spectra[layer_idx] = {}

        for name, module in model.named_modules():
            if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
                continue
            if f'layers.{layer_idx}.' not in name:
                continue

            for adapter_name in module.lora_A:
                A = module.lora_A[adapter_name].weight.detach().float()  # (r, in)
                B = module.lora_B[adapter_name].weight.detach().float()  # (out, r)

                scaling = (module.scaling[adapter_name]
                           if isinstance(module.scaling, dict)
                           else module.scaling)

                # 有效权重 ΔW = (B @ A) * scaling
                delta_W = (B @ A).cpu().numpy() * scaling  # (out, in)

                # SVD 分解
                # delta_W (out × in), full_matrices=False → U(out×k), S(k,), Vt(k×in)
                # k = min(out, in), 但实际秩 ≤ r
                U, S, Vt = np.linalg.svd(delta_W, full_matrices=False)

                # 只保留前 r 个（LoRA 秩）
                r = A.shape[0]
                U = U[:, :r]
                S = S[:r]
                Vt = Vt[:r, :]

                module_short_name = name.split(f'layers.{layer_idx}.')[-1]

                spectra[layer_idx][module_short_name] = {
                    "singular_values": S,
                    "U": U,
                    "V": Vt.T,          # V (in × r)
                    "delta_W": delta_W,
                    "rank": r,
                    "scaling": scaling,
                }

    return spectra


# ============================================================================
# 2. 谱指标计算
# ============================================================================

def compute_spectral_metrics(
    spectra: Dict[int, Dict[str, Dict]],
) -> Dict[int, Dict[str, float]]:
    """
    为每层计算谱指标（跨该层所有 LoRA 模块聚合）。

    指标:
      - spectral_norm:    σ_max（最大奇异值），最大增益
      - nuclear_norm:     Σσᵢ（核范数），总能量
      - frobenius_norm:   √(Σσᵢ²)（Frobenius 范数）
      - effective_rank:   exp(熵)，信息分散度
      - spectral_entropy: -Σ pᵢ log pᵢ，谱的均匀性
      - participation_ratio: (Σσᵢ²)² / Σσᵢ⁴，有效维度
      - condition_number:  σ_max / σ_min，数值稳定性
      - top1_ratio:        σ₁² / Σσᵢ²，能量集中度
      - top2_ratio:        (σ₁²+σ₂²) / Σσᵢ²
    """
    metrics = {}

    for layer_idx, modules in spectra.items():
        all_sv = []  # 收集该层所有模块的奇异值
        all_sv_sq = []

        spectral_norms = []
        nuclear_norms = []
        frob_norms = []
        eff_ranks = []
        entropies = []
        prs = []
        cond_nums = []
        top1_ratios = []
        top2_ratios = []

        for mod_name, mod_data in modules.items():
            S = mod_data["singular_values"]
            S = S[S > 1e-12]  # 过滤数值零

            if len(S) == 0:
                continue

            S_sq = S ** 2

            # 基本范数
            spectral_norms.append(S[0])
            nuclear_norms.append(np.sum(S))
            frob_norms.append(np.sqrt(np.sum(S_sq)))

            # 有效秩: exp(熵)
            p = S / np.sum(S)
            entropy = -np.sum(p * np.log(p + 1e-12))
            eff_rank = np.exp(entropy)
            eff_ranks.append(eff_rank)
            entropies.append(entropy)

            # 参与率
            pr = (np.sum(S_sq) ** 2) / (np.sum(S_sq ** 2) + 1e-12)
            prs.append(pr)

            # 条件数
            cond = S[0] / (S[-1] + 1e-12)
            cond_nums.append(cond)

            # 能量集中度
            total_energy = np.sum(S_sq)
            top1_ratios.append(S_sq[0] / total_energy)
            if len(S) >= 2:
                top2_ratios.append((S_sq[0] + S_sq[1]) / total_energy)
            else:
                top2_ratios.append(1.0)

            all_sv.extend(S.tolist())
            all_sv_sq.extend(S_sq.tolist())

        # 聚合（取该层所有模块的均值）
        metrics[layer_idx] = {
            "spectral_norm": float(np.mean(spectral_norms)),
            "nuclear_norm": float(np.mean(nuclear_norms)),
            "frobenius_norm": float(np.mean(frob_norms)),
            "effective_rank": float(np.mean(eff_ranks)),
            "spectral_entropy": float(np.mean(entropies)),
            "participation_ratio": float(np.mean(prs)),
            "condition_number": float(np.mean(cond_nums)),
            "top1_energy_ratio": float(np.mean(top1_ratios)),
            "top2_energy_ratio": float(np.mean(top2_ratios)),
            "num_modules": len(modules),
        }

    return metrics


# ============================================================================
# 3. 可视化
# ============================================================================

def plot_spectral_analysis(
    spectra: Dict[int, Dict[str, Dict]],
    metrics: Dict[int, Dict[str, float]],
    harmful_layers: List[int] = [0, 1, 2, 3, 11],
    important_layers: List[int] = [5, 13, 14, 15, 16, 28],
    output_dir: str = "./results/spectral_analysis",
):
    """绘制 4 张图：谱形态、谱指标对比、能量分布、层级热力图"""

    os.makedirs(output_dir, exist_ok=True)

    # ---- 图 1: 有害层 vs 重要层的奇异值分布 ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    for layer_idx in harmful_layers:
        if layer_idx not in spectra:
            continue
        # 取该层所有模块的奇异值均值
        all_sv = []
        for mod_name, mod_data in spectra[layer_idx].items():
            sv = mod_data["singular_values"]
            all_sv.append(sv)
        mean_sv = np.mean(all_sv, axis=0)
        ax1.plot(range(1, len(mean_sv) + 1), mean_sv, 'o-',
                 label=f'Layer {layer_idx} (harmful)', linewidth=2, markersize=8)

    ax1.set_xlabel('Singular Value Index', fontsize=13)
    ax1.set_ylabel('Singular Value', fontsize=13)
    ax1.set_title('Harmful Layers: Singular Value Spectrum', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.set_xticks(range(1, 9))
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for layer_idx in important_layers:
        if layer_idx not in spectra:
            continue
        all_sv = []
        for mod_name, mod_data in spectra[layer_idx].items():
            sv = mod_data["singular_values"]
            all_sv.append(sv)
        mean_sv = np.mean(all_sv, axis=0)
        ax2.plot(range(1, len(mean_sv) + 1), mean_sv, 's-',
                 label=f'Layer {layer_idx} (important)', linewidth=2, markersize=8)

    ax2.set_xlabel('Singular Value Index', fontsize=13)
    ax2.set_ylabel('Singular Value', fontsize=13)
    ax2.set_title('Important Layers: Singular Value Spectrum', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.set_xticks(range(1, 9))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(output_dir, "spectrum_comparison.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path1}")

    # ---- 图 2: 6 个谱指标 × 32 层柱状图 ----
    metric_names = [
        ("spectral_norm", "Spectral Norm (σ_max)"),
        ("effective_rank", "Effective Rank"),
        ("spectral_entropy", "Spectral Entropy"),
        ("top1_energy_ratio", "Top-1 Energy Ratio"),
        ("participation_ratio", "Participation Ratio"),
        ("condition_number", "Condition Number (log scale)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()

    for ax_idx, (metric_key, metric_title) in enumerate(metric_names):
        ax = axes[ax_idx]
        layer_indices = sorted(metrics.keys())
        values = [metrics[ℓ][metric_key] for ℓ in layer_indices]

        colors = []
        for ℓ in layer_indices:
            if ℓ in harmful_layers:
                colors.append('#e74c3c')    # 红色 = 有害层
            elif ℓ in important_layers:
                colors.append('#2ecc71')    # 绿色 = 重要层
            else:
                colors.append('#95a5a6')    # 灰色 = 普通

        bars = ax.bar(layer_indices, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(metric_title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer Index', fontsize=11)
        ax.set_ylabel(metric_title, fontsize=11)
        ax.set_xticks(range(0, 32, 2))
        ax.grid(True, alpha=0.2, axis='y')

        if 'condition_number' in metric_key:
            ax.set_yscale('log')

        # 标注有害层和重要层
        for ℓ in harmful_layers:
            if ℓ in layer_indices:
                idx = layer_indices.index(ℓ)
                ax.annotate(f'L{ℓ}', (ℓ, values[idx]),
                           textcoords="offset points", xytext=(0, 5),
                           fontsize=8, color='#e74c3c', fontweight='bold')
        for ℓ in important_layers:
            if ℓ in layer_indices:
                idx = layer_indices.index(ℓ)
                ax.annotate(f'L{ℓ}', (ℓ, values[idx]),
                           textcoords="offset points", xytext=(0, 5),
                           fontsize=8, color='#2ecc71', fontweight='bold')

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Harmful (pruned)'),
        Patch(facecolor='#2ecc71', label='Important (kept)'),
        Patch(facecolor='#95a5a6', label='Normal'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11)

    plt.suptitle('Spectral Metrics Across 32 Layers', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path2 = os.path.join(output_dir, "spectral_metrics_32layers.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path2}")

    # ---- 图 3: 全 32 层奇异值热力图 ----
    fig, ax = plt.subplots(figsize=(16, 8))

    # 构建 (32, r) 矩阵：每行是一层的平均奇异值
    all_mean_sv = []
    for layer_idx in range(32):
        if layer_idx not in spectra:
            all_mean_sv.append(np.zeros(8))
            continue
        all_sv = []
        for mod_data in spectra[layer_idx].values():
            sv = mod_data["singular_values"]
            all_sv.append(sv)
        if all_sv:
            mean_sv = np.mean(all_sv, axis=0)
            # pad to 8 if needed
            if len(mean_sv) < 8:
                mean_sv = np.pad(mean_sv, (0, 8 - len(mean_sv)))
            all_mean_sv.append(mean_sv)
        else:
            all_mean_sv.append(np.zeros(8))

    sv_matrix = np.array(all_mean_sv)  # (32, 8)

    im = ax.imshow(sv_matrix, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Singular Value Index', fontsize=13)
    ax.set_ylabel('Layer Index', fontsize=13)
    ax.set_title('Singular Value Heatmap (all 32 layers)', fontsize=14)
    ax.set_xticks(range(8))
    ax.set_yticks(range(0, 32, 2))
    plt.colorbar(im, ax=ax, label='Singular Value')

    # 标注有害层
    for ℓ in harmful_layers:
        ax.axhline(y=ℓ - 0.5, color='cyan', linewidth=1.5, linestyle='--', alpha=0.7)
        ax.text(8.3, ℓ, f'L{ℓ}', fontsize=8, color='cyan', va='center')
    for ℓ in important_layers:
        ax.axhline(y=ℓ - 0.5, color='lime', linewidth=1.5, linestyle='--', alpha=0.7)
        ax.text(8.3, ℓ, f'L{ℓ}', fontsize=8, color='lime', va='center')

    plt.tight_layout()
    path3 = os.path.join(output_dir, "singular_value_heatmap.png")
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path3}")

    # ---- 图 4: 有害 vs 重要 的指标统计对比（箱线图） ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax_idx, (metric_key, metric_title) in enumerate(metric_names):
        ax = axes[ax_idx]

        harmful_vals = [metrics[ℓ][metric_key] for ℓ in harmful_layers if ℓ in metrics]
        important_vals = [metrics[ℓ][metric_key] for ℓ in important_layers if ℓ in metrics]
        normal_layers = [ℓ for ℓ in range(32) if ℓ not in harmful_layers and ℓ not in important_layers]
        normal_vals = [metrics[ℓ][metric_key] for ℓ in normal_layers if ℓ in metrics]

        bp = ax.boxplot(
            [harmful_vals, important_vals, normal_vals],
            labels=['Harmful', 'Important', 'Normal'],
            patch_artist=True,
            widths=0.6,
        )
        bp['boxes'][0].set_facecolor('#e74c3c')
        bp['boxes'][1].set_facecolor('#2ecc71')
        bp['boxes'][2].set_facecolor('#95a5a6')

        ax.set_title(metric_title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle('Spectral Metrics: Harmful vs Important vs Normal',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path4 = os.path.join(output_dir, "metrics_boxplot.png")
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path4}")


# ============================================================================
# 4. 主流程
# ============================================================================

def run_spectral_analysis(
    base_model_path: str,
    lora_adapter_path: str,
    num_layers: int = 32,
    harmful_layers: List[int] = [0, 1, 2, 3, 11],
    important_layers: List[int] = [5, 13, 14, 15, 16, 28],
    output_dir: str = "./results/spectral_analysis",
):
    """完整谱分析流程"""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("LoRA Adapter Spectral Analysis")
    print(f"Model: {base_model_path}")
    print(f"LoRA:  {lora_adapter_path}")
    print(f"Harmful layers:  {harmful_layers}")
    print(f"Important layers: {important_layers}")
    print("=" * 70)

    # === Step 1: 加载模型 ===
    print("\nLoading model...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, lora_adapter_path)

    # === Step 2: 计算 SVD 谱 ===
    print("\nComputing SVD spectra for all 32 layers...")
    spectra = compute_layer_spectrum(model, num_layers)
    print(f"  Done. {len(spectra)} layers analyzed.")

    # === Step 3: 计算谱指标 ===
    print("\nComputing spectral metrics...")
    metrics = compute_spectral_metrics(spectra)

    # === Step 4: 打印对比表 ===
    print("\n" + "=" * 120)
    print(f"{'Layer':>6} | {'Type':>10} | {'σ_max':>10} | {'σ_sum':>10} | "
          f"{'eff_rank':>10} | {'entropy':>10} | {'top1_ratio':>10} | "
          f"{'part_ratio':>10} | {'cond_num':>10}")
    print("-" * 120)

    all_layers = sorted(metrics.keys())
    for layer_idx in all_layers:
        m = metrics[layer_idx]
        if layer_idx in harmful_layers:
            layer_type = "HARMFUL"
        elif layer_idx in important_layers:
            layer_type = "IMPORTANT"
        else:
            layer_type = "normal"

        print(f"{layer_idx:>6} | {layer_type:>10} | {m['spectral_norm']:>10.6f} | "
              f"{m['nuclear_norm']:>10.6f} | {m['effective_rank']:>10.4f} | "
              f"{m['spectral_entropy']:>10.4f} | {m['top1_energy_ratio']:>10.4f} | "
              f"{m['participation_ratio']:>10.4f} | {m['condition_number']:>10.2f}")

    print("=" * 120)

    # === Step 5: 统计对比 ===
    print("\n--- Group Statistics ---")
    for group_name, group_layers in [("Harmful", harmful_layers), ("Important", important_layers)]:
        group_metrics = {k: [metrics[ℓ][k] for ℓ in group_layers if ℓ in metrics]
                        for k in metrics[group_layers[0]].keys()
                        if k != "num_modules"}
        print(f"\n  {group_name} layers ({group_layers}):")
        for metric_name, vals in group_metrics.items():
            print(f"    {metric_name:>25}: mean={np.mean(vals):.6f}, std={np.std(vals):.6f}")

    # === Step 6: 可视化 ===
    print("\nGenerating plots...")
    plot_spectral_analysis(
        spectra, metrics,
        harmful_layers=harmful_layers,
        important_layers=important_layers,
        output_dir=output_dir,
    )

    # === Step 7: 保存结果 ===
    # 保存奇异值（用于后续分析）
    sv_data = {}
    for layer_idx, modules in spectra.items():
        sv_data[layer_idx] = {}
        for mod_name, mod_data in modules.items():
            sv_data[layer_idx][mod_name] = {
                "singular_values": mod_data["singular_values"].tolist(),
                "scaling": mod_data["scaling"],
            }

    sv_path = os.path.join(output_dir, "singular_values.json")
    with open(sv_path, "w") as f:
        json.dump(sv_data, f, indent=2)
    print(f"\nSingular values saved to: {sv_path}")

    # 保存指标
    metrics_path = os.path.join(output_dir, "spectral_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({str(k): v for k, v in metrics.items()}, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # === Step 8: 打印关键发现 ===
    print("\n" + "=" * 70)
    print("Key Findings")
    print("=" * 70)

    # 对比有害层和重要层的各指标均值
    h_metrics = {k: np.mean([metrics[ℓ][k] for ℓ in harmful_layers if ℓ in metrics])
                 for k in ["spectral_norm", "effective_rank", "spectral_entropy",
                           "top1_energy_ratio", "participation_ratio", "condition_number"]}
    i_metrics = {k: np.mean([metrics[ℓ][k] for ℓ in important_layers if ℓ in metrics])
                 for k in ["spectral_norm", "effective_rank", "spectral_entropy",
                           "top1_energy_ratio", "participation_ratio", "condition_number"]}

    print(f"\n  {'Metric':>25} | {'Harmful (mean)':>15} | {'Important (mean)':>15} | {'Direction':>12}")
    print("  " + "-" * 75)
    for k in h_metrics:
        direction = "↑ harmful" if h_metrics[k] > i_metrics[k] else "↓ important"
        print(f"  {k:>25} | {h_metrics[k]:>15.6f} | {i_metrics[k]:>15.6f} | {direction:>12}")

    print("\nSpectral analysis complete!")

    # 清理
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return spectra, metrics


# ============================================================================
# 5. Entry Point
# ============================================================================

if __name__ == "__main__":
    spectra, metrics = run_spectral_analysis(
        base_model_path="/root/autodl-fs/llama-7b",
        lora_adapter_path="./llama-logic-sft-final",
        num_layers=32,
        harmful_layers=[0, 1, 2, 3, 11],           # 实验确认的有害层
        important_layers=[5, 13, 14, 15, 16, 28],   # 实验确认的重要层
        output_dir="./results/spectral_analysis_b11",
    )