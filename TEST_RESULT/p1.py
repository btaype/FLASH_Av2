import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print(os.getcwd())

csv_path = r"flash2_local_results.csv"
df = pd.read_csv(csv_path)

seqs = [512, 1024, 2048, 4096, 8192, 16384]
labels = ["512", "1k", "2k", "4k", "8k", "16k"]

# Valores del paper A100 aproximados desde tus gráficas
paper = {
    "fwd_bwd": {
        (False, 64):  [132, 162, 162, 171, 175, 176],
        (False, 128): [150, 172, 187, 196, 201, 203],
        (True, 64):   [88, 119, 140, 156, 165, 171],
        (True, 128):  [99, 133, 155, 173, 182, 189],
    },
    "fwd": {
        (False, 64):  [191, 193, 193, 192, 192, 192],
        (False, 128): [221, 226, 227, 222, 224, 223],
        (True, 64):   [115, 146, 167, 177, 181, 183],
        (True, 128):  [132, 164, 187, 198, 200, 197],
    }
}

def get_local_values(causal, headdim, metric):
    col = "fwd_bwd_tflops" if metric == "fwd_bwd" else "fwd_tflops"
    sub = df[(df["causal"] == causal) & (df["headdim"] == headdim)]
    vals = []
    for s in seqs:
        row = sub[sub["seqlen"] == s]
        vals.append(float(row[col].iloc[0]) if len(row) else np.nan)
    return vals

def plot_figure(metric, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    configs = [
        (False, 64,  "(a) Without causal mask, head dimension 64"),
        (False, 128, "(b) Without causal mask, head dimension 128"),
        (True, 64,   "(c) With causal mask, head dimension 64"),
        (True, 128,  "(d) With causal mask, head dimension 128"),
    ]

    for ax, (causal, headdim, subtitle) in zip(axes, configs):
        x = np.arange(len(seqs))
        width = 0.35

        a100_vals = paper[metric][(causal, headdim)]
        local_vals = get_local_values(causal, headdim, metric)

        ax.bar(x - width/2, a100_vals, width, label="FlashAttention-2 (A100)")
        ax.bar(x + width/2, local_vals, width, label="FlashAttention-2 (RTX 4060 Ti)")

        for i, v in enumerate(a100_vals):
            ax.text(i - width/2, v + 3, f"{v:.0f}", ha="center", fontsize=8)

        for i, v in enumerate(local_vals):
            if not np.isnan(v):
                ax.text(i + width/2, v + 3, f"{v:.1f}", ha="center", fontsize=8)

        ax.set_title(subtitle, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Speed (TFLOPs/s)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

plot_figure(
    "fwd_bwd",
    "Attention forward + backward speed: A100 vs RTX 4060 Ti",
    "figure_forward_backward.png"
)

plot_figure(
    "fwd",
    "Attention forward speed: A100 vs RTX 4060 Ti",
    "figure_forward.png"
)

print("Listo: figure_forward_backward.png y figure_forward.png")