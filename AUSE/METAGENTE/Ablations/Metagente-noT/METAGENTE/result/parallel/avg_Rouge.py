# rouge_sbert_bertscore_mean_std.py
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from bert_score import score as bertscore_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util

# =============== CONFIG ===============
FILES: List[str] = [
    "/Users/Yuki/Prj_multi_agent_git/Metagente-noT/METAGENTE/result/parallel/noT_600.csv",
    "/Users/Yuki/Prj_multi_agent_git/Metagente-noT/METAGENTE/result/parallel/noT2_600.csv",
    "/Users/Yuki/Prj_multi_agent_git/Metagente-noT/METAGENTE/result/parallel/noT3_600.csv",
]

# Các tên cột khả dĩ
REF_CANDIDATES = ["description_short", "Description"]
PRED_CANDIDATES = ["generated_about", "Generated About"]

# SBERT & BERTScore
SBERT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BERTSCORE_MODEL = "microsoft/deberta-large-mnli"
BERTSCORE_LANG = None
BERTSCORE_BATCH_SIZE = 64
# =====================================


def _safe_text(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip()


def _find_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Không tìm thấy bất kỳ cột nào trong {candidates}. Columns: {list(df.columns)}"
    )


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    a = np.array(xs, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(a)), float(np.std(a, ddof=1))


def compute_file_metrics(
    csv_path: str,
    sbert_model: SentenceTransformer,
    device: torch.device,
) -> Tuple[Dict[str, Tuple[float, float]], int, str, str]:
    """
    Trả về dict metric -> (mean, std)
    """
    df = pd.read_csv(csv_path)
    ref_col = _find_existing_column(df, REF_CANDIDATES)
    pred_col = _find_existing_column(df, PRED_CANDIDATES)

    refs, hyps = [], []
    for _, row in df.iterrows():
        r = _safe_text(row[ref_col])
        h = _safe_text(row[pred_col])
        if r == "" and h == "":
            continue
        refs.append(r)
        hyps.append(h)

    n_valid = len(refs)
    if n_valid == 0:
        return {}, 0, ref_col, pred_col

    # ----- ROUGE -----
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for r, h in zip(refs, hyps):
        sc = rouge.score(r, h)
        r1.append(sc["rouge1"].fmeasure)
        r2.append(sc["rouge2"].fmeasure)
        rL.append(sc["rougeL"].fmeasure)

    # ----- SBERT -----
    with torch.inference_mode():
        ref_emb = sbert_model.encode(
            refs, convert_to_tensor=True, device=device, normalize_embeddings=True
        )
        hyp_emb = sbert_model.encode(
            hyps, convert_to_tensor=True, device=device, normalize_embeddings=True
        )
        sims = st_util.cos_sim(ref_emb, hyp_emb).diagonal().tolist()

    # ----- BERTScore -----
    P, R, F1 = bertscore_score(
        cands=hyps,
        refs=refs,
        model_type=BERTSCORE_MODEL,
        lang=BERTSCORE_LANG,
        device=str(device) if device.type == "cuda" else "cpu",
        batch_size=BERTSCORE_BATCH_SIZE,
        rescale_with_baseline=False,
        verbose=False,
    )
    P, R, F1 = [float(x) for x in P], [float(x) for x in R], [float(x) for x in F1]

    metrics = {
        "ROUGE-1": _mean_std(r1),
        "ROUGE-2": _mean_std(r2),
        "ROUGE-L": _mean_std(rL),
        "SBERT": _mean_std(sims),
        "BERTScore_P": _mean_std(P),
        "BERTScore_R": _mean_std(R),
        "BERTScore_F1": _mean_std(F1),
    }
    return metrics, n_valid, ref_col, pred_col


def _combine_mean_std(
    file_metrics: List[Dict[str, Tuple[float, float]]],
) -> Dict[str, Tuple[float, float]]:
    """Tính mean ± std giữa các file (macro-level)"""
    keys = list(file_metrics[0].keys())
    macro = {}
    for k in keys:
        vals = [m[k][0] for m in file_metrics if k in m and not math.isnan(m[k][0])]
        macro[k] = (
            (float(np.mean(vals)), float(np.std(vals, ddof=1)))
            if vals
            else (float("nan"), float("nan"))
        )
    return macro


def _fmt(mean: float, std: float) -> str:
    if math.isnan(mean) or math.isnan(std):
        return "nan"
    return f"{mean:.3f} ± {std:.3f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=str(device))

    per_file_rows = []
    file_metrics_list = []

    for f in FILES:
        mets, n, ref_col, pred_col = compute_file_metrics(f, sbert_model, device)
        file_metrics_list.append(mets)
        row = {"file": f, "ref_col": ref_col, "pred_col": pred_col, "n_pairs": n}
        for k, (m, s) in mets.items():
            row[k] = _fmt(m, s)
        per_file_rows.append(row)

    # Macro-level mean ± std
    macro = _combine_mean_std(file_metrics_list)
    overall = {
        "file": "MACRO across files",
        "ref_col": "-",
        "pred_col": "-",
        "n_pairs": sum(r["n_pairs"] for r in per_file_rows),
    }
    for k, (m, s) in macro.items():
        overall[k] = _fmt(m, s)

    df_out = pd.DataFrame(per_file_rows + [overall])
    print(df_out.to_string(index=False))

    out_path = "/Users/Yuki/Prj_multi_agent_git/Metagente-noT/METAGENTE/result/parallel/rouge_sbert_bertscore_mean_std.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nĐã lưu: {out_path}")


if __name__ == "__main__":
    main()
