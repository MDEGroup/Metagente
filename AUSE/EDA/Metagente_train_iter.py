import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILES = {
    "Origin_LLM_set": "Origin_LLM_set.json",
    "Mistral": "Mistral.json",
    "Llama": "Llama.json",
    "Gemma": "gemma.json",
    "GPT-3.5": "gpt-3.5-turbo.json",
}
OUT_DIR = Path("./viz_out_llm_avg")
FORCE_COMMON_LAST_ITER = 7


def load_json_train_debug(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("Train Debug", [])


def mean_curve_from_json(train_debug, metric_key: str):
    seqs = []
    for sample in train_debug:
        iters = sample.get("iteration_debug", []) or []
        seq = [it.get(metric_key, np.nan) for it in iters]
        if any(isinstance(v, (int, float)) and np.isfinite(v) for v in seq):
            seqs.append([np.nan if v is None else float(v) for v in seq])
    if not seqs:
        return None
    max_len = max(len(s) for s in seqs)
    arr = np.full((len(seqs), max_len), np.nan, dtype=float)
    for r, s in enumerate(seqs):
        arr[r, : len(s)] = s
    return np.nanmean(arr, axis=0)


def mean_curve_from_csv_turbo(
    path: Path,
    r1="ROUGE-1 score",
    r2="ROUGE-2 score",
    rl="ROUGE-L score",
    cid="Data ID",
    cit="Iteration",
):
    df = pd.read_csv(path)

    def pick(candidates):
        for c in [candidates] if isinstance(candidates, str) else candidates:
            if c in df.columns:
                return c
        raise KeyError(f"Missing columns: {candidates}")

    cid = pick([cid, "sample_id", "Sample ID"])
    cit = pick([cit, "iteration", "Iter"])
    r1 = pick([r1, "rouge1_score", "Rouge1 Score"])
    r2 = pick([r2, "rouge2_score", "Rouge2 Score"])
    rl = pick([rl, "rougeL_score", "RougeL Score", "ROUGE-L"])

    df = df.dropna(subset=[r1, r2, rl], how="all")

    def make_curve(score_col):
        seqs = []
        for _, g in df.groupby(cid):
            seq = g.sort_values(by=cit)[score_col].to_numpy(dtype=float)
            if np.isfinite(seq).any():
                seqs.append(seq)
        if not seqs:
            return None
        max_len = max(len(s) for s in seqs)
        arr = np.full((len(seqs), max_len), np.nan)
        for r, s in enumerate(seqs):
            arr[r, : len(s)] = s
        return np.nanmean(arr, axis=0)

    return {
        "R1": make_curve(r1),
        "R2": make_curve(r2),
        "L": make_curve(rl),
    }


def load_all_models(files: dict[str, str]):
    curves_by_model = {}
    for name, path in files.items():
        p = Path(path)
        if not p.exists():
            print(f"[warn] Missing file for {name}: {p}")
            continue
        if p.suffix.lower() == ".json":
            td = load_json_train_debug(p)
            curves_by_model[name] = {
                "R1": mean_curve_from_json(td, "rouge1_score"),
                "R2": mean_curve_from_json(td, "rouge2_score"),
                "L": mean_curve_from_json(td, "rougeL_score"),
            }
        else:
            curves_by_model[name] = mean_curve_from_csv_turbo(p)
    return curves_by_model


def truncate_to_common(curves_by_model, metric: str, force_last_iter=None):
    lengths = []
    for m, d in curves_by_model.items():
        y = (d or {}).get(metric)
        if y is not None and len(y) > 0:
            lengths.append(len(y))
    if not lengths:
        return {}, 0
    common_len = min(lengths)
    if force_last_iter is not None:
        common_len = min(common_len, int(force_last_iter) + 1)
        if common_len <= 0:
            return {}, 0
    out = {}
    for m, d in curves_by_model.items():
        y = (d or {}).get(metric)
        if y is None or len(y) < common_len:
            continue
        out[m] = y[:common_len]
    return out, common_len


def save_csv(curves_trunc, metric, out_dir):
    rows = []
    for model, y in curves_trunc.items():
        for i, v in enumerate(y):
            rows.append(
                {"model": model, "metric": metric, "iteration": i, "mean_score": v}
            )
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / f"avg_curve_{metric}_fair.csv", index=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curves_by_model = load_all_models(FILES)

    metric_labels = {"R1": "ROUGE-1", "R2": "ROUGE-2", "L": "ROUGE-L"}
    for metric in ["R1", "R2", "L"]:
        curves_trunc, common_len = truncate_to_common(
            curves_by_model, metric, force_last_iter=FORCE_COMMON_LAST_ITER
        )
        if common_len == 0 or len(curves_trunc) == 0:
            print(f"[skip] No common window for {metric}")
            continue

        save_csv(curves_trunc, metric, OUT_DIR)

        x = np.arange(common_len)
        plt.figure()
        for model, y in curves_trunc.items():
            plt.plot(x, y, marker="o", label=model)
        plt.title(
            f"Average Learning Curve — {metric_labels[metric]} (fair up to iter {common_len - 1})"
        )
        plt.xlabel("Iteration")
        plt.ylabel(metric_labels[metric])
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"avg_curve_{metric}_fair.png", dpi=160)
        plt.close()

    print("Done. Check:", OUT_DIR.resolve())
    if FORCE_COMMON_LAST_ITER is None:
        print("Common window auto-picked = min length across models per metric.")
    else:
        print(f"Forced common last iter = {FORCE_COMMON_LAST_ITER}")


if __name__ == "__main__":
    main()
