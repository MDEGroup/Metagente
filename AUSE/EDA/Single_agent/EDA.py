from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_CSV = Path("score/merged_summaries_600_with_html_and_rouge.csv")
OUT_DIR = Path("./viz_out_violin")
OUTPUT_PNG = "single_vs_metagente_600_violin_box.png"

# Column mapping -> display name
# (columns look like: "<Model>_ROUGE-1", "<Model>_ROUGE-2", "<Model>_ROUGE-L")
MODEL_NAME_MAP = {
    "Llama": "Llama",
    "Mistral": "Mistral",
    "GPT-4o": "Gpt-4o",
    "OG": "Metagente",
    "Gemma": "Gemma",
}
ROUGE_COLS = {"1": "ROUGE-1", "2": "ROUGE-2", "L": "ROUGE-L"}


def load_scores_from_merged(df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """
    Returns all_scores[display_name][metric_key] = np.ndarray
    metric_key ∈ {"1","2","L"}
    """
    all_scores: dict[str, dict[str, np.ndarray]] = {}
    for model_col, display_name in MODEL_NAME_MAP.items():
        per_metric: dict[str, np.ndarray] = {}
        for k, rouge_name in ROUGE_COLS.items():
            col = f"{model_col}_{rouge_name}"
            if col not in df.columns:
                per_metric[k] = np.array([], dtype=float)
                continue
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            arr = np.clip(arr, 0.0, 1.0)
            per_metric[k] = arr
        all_scores[display_name] = per_metric
    return all_scores


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        print(f"[error] File not found: {INPUT_CSV.resolve()}")
        return

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 15,
            "ytick.labelsize": 14,
            "legend.fontsize": 18,
            "figure.titlesize": 18,
        }
    )

    df = pd.read_csv(INPUT_CSV)
    all_scores = load_scores_from_merged(df)

    models = list(MODEL_NAME_MAP.values())
    metrics = ["1", "2", "L"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax, m in zip(axes, metrics):
        data = [all_scores[md][m] for md in models]
        valid_idx = [i for i, arr in enumerate(data) if arr.size > 0]
        if not valid_idx:
            ax.set_title(f"ROUGE-{m}: no data", fontweight="bold")
            ax.axis("off")
            continue

        data = [data[i] for i in valid_idx]
        labels = [models[i] for i in valid_idx]
        positions = np.arange(1, len(data) + 1)
        sub_colors = [colors[i] for i in valid_idx]

        parts = ax.violinplot(
            data,
            positions=positions,
            showmeans=False,
            showmedians=True,
            widths=0.85,
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(sub_colors[i])
            pc.set_alpha(0.4)
        if "cmedians" in parts:
            parts["cmedians"].set_linewidth(2)

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.25,
            patch_artist=True,
            showfliers=False,
        )
        for patch, c in zip(bp["boxes"], sub_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        for line in bp["whiskers"] + bp["caps"] + bp["medians"]:
            line.set_linewidth(1.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(f"ROUGE-{m}", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("ROUGE Score")
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Single agents vs Metagente (600 samples)", y=0.995, fontweight="bold")

    handles = [
        plt.Line2D([0], [0], color=colors[i], lw=8, alpha=0.6)
        for i in range(len(models))
    ]
    fig.legend(handles, models, loc="lower center", ncol=len(models))

    fig.tight_layout(rect=[0, 0.15, 1, 0.93])

    out_png = OUT_DIR / OUTPUT_PNG
    plt.savefig(out_png, dpi=160)
    plt.close()
    print("Saved:", out_png)


if __name__ == "__main__":
    main()
