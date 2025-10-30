import html
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from utils.llm import OpenAIClient


# -----------------------------
# Text helpers
# -----------------------------
def strip_html_tags(s: str) -> str:
    if s is None:
        return ""
    s = html.unescape(s)
    return re.sub(r"<[^>]+>", "", s)


def whitespace_norm(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def light_clean_generation(s: str) -> str:
    s = s or ""
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        s = s[1:-1].strip()
    s = re.sub(r"^```+\s*", "", s)
    s = re.sub(r"\s*```+$", "", s)
    s = whitespace_norm(s)
    return s


# -----------------------------
# ROUGE-L (recall)
# -----------------------------
def _tok(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", s or "")


def _lcs(a: List[str], b: List[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rougeL_recall(ref: str, hyp: str) -> float:
    r = _tok(ref)
    h = _tok(hyp)
    if not r:
        return 0.0
    lcs = _lcs(r, h)
    return lcs / (len(r) + 1e-9)


# -----------------------------
# Split function
# -----------------------------
def split_first_second(
    raw_desc: str, ratio_low: float = 0.5, ratio_high: float = 0.6
) -> Tuple[str, str]:
    """Cắt first_piece (≈50–60%) và phần còn lại."""
    raw_desc = raw_desc or ""
    raw_desc = raw_desc.strip()
    if not raw_desc:
        return "", ""

    sentences = re.split(r"(?<=[\.\!\?\;])\s+|\n+", raw_desc)
    sentences = [s.strip() for s in sentences if s.strip()]

    toks = _tok(raw_desc)
    total = max(1, len(toks))
    target = int(total * random.uniform(ratio_low, ratio_high))

    if len(sentences) > 1:
        cum, first_parts = 0, []
        for s in sentences:
            cum += len(_tok(s))
            first_parts.append(s)
            if cum >= target:
                break
        first_piece = " ".join(first_parts).strip()
        second_piece = raw_desc[len(first_piece) :].strip()
        if len(_tok(second_piece)) < 3:
            return _split_by_token(raw_desc, target)
        return first_piece, second_piece
    else:
        return _split_by_token(raw_desc, target)


def _split_by_token(text: str, target_tokens: int) -> Tuple[str, str]:
    toks = _tok(text)
    if len(toks) < 6:
        return text, ""
    words = re.findall(r"\S+\s*", text)
    acc, count = [], 0
    for w in words:
        acc.append(w)
        count += len(_tok(w))
        if count >= target_tokens:
            break
    first_piece = "".join(acc).strip()
    second_piece = text[len(first_piece) :].strip()
    return first_piece, second_piece


# -----------------------------
# Prompts
# -----------------------------
GUIDED_TMPL = """You are given the first part of an app description from the AndroZoo dataset, which contains metadata collected from Google Play.

Please complete the remaining part of the description exactly as it appears in the dataset, without adding or removing information.

---
App description (first part):
{first_piece}

Continue the description (second part):
"""

GENERAL_TMPL = """You are given the beginning of an app description.

Please continue writing the next part of the description naturally so that it forms a complete and coherent text, consistent in style and meaning.

---
App description (first part):
{first_piece}

Continue the description (second part):
"""


# -----------------------------
# LLM Wrapper
# -----------------------------
@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    max_tokens: int = 800
    temperature: float = 0.0


class LLMWrapper:
    def __init__(self, cfg: LLMConfig):
        self.llm = OpenAIClient(model=cfg.model)
        self.cfg = cfg

    def call(self, prompt: str) -> str:
        try:
            text = self.llm.generate(
                prompt, max_token=self.cfg.max_tokens, temperature=self.cfg.temperature
            )
            if isinstance(text, dict) and "text" in text:
                return str(text["text"])
            return str(text)
        except Exception as e:
            return f"[LLM_ERROR] {e}"


# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    csv_path = "/Users/Yuki/Prj_multi_agent_git/Metagente/METAGENTE/data/test_data.csv"

    out_csv = "results_androozoo_contam_full_600.csv"
    random.seed(42)

    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    if "description_html" not in df.columns:
        raise ValueError("CSV phải có cột 'description_html'.")

    df["_desc"] = df["description_html"].astype(str).str.strip()
    # df = df[df["_desc"].str.len() > 30].copy()
    n_samples = len(df)
    print(f"Total valid samples: {n_samples}")

    llm = LLMWrapper(LLMConfig(model="gpt-4o-mini"))

    rows: List[Dict[str, Any]] = []
    for i, row in tqdm(
        df.reset_index(drop=True).iterrows(), total=len(df), desc="Processing"
    ):
        raw_desc = strip_html_tags(row["_desc"])
        first_piece, second_piece = split_first_second(raw_desc)
        first_piece = whitespace_norm(first_piece)
        second_piece = whitespace_norm(second_piece)
        if len(first_piece) < 15 or len(second_piece) < 5:
            continue

        guided_prompt = GUIDED_TMPL.format(first_piece=first_piece)
        general_prompt = GENERAL_TMPL.format(first_piece=first_piece)

        guided_out = llm.call(guided_prompt)
        general_out = llm.call(general_prompt)
        guided_out = light_clean_generation(guided_out)
        general_out = light_clean_generation(general_out)

        rL_guided = rougeL_recall(second_piece, guided_out)
        rL_general = rougeL_recall(second_piece, general_out)
        delta = rL_guided - rL_general

        rows.append(
            {
                "idx": i,
                "first_piece": first_piece,
                "reference_second_piece": second_piece,
                "guided_output": guided_out,
                "general_output": general_out,
                "rougeL_guided": round(rL_guided, 4),
                "rougeL_general": round(rL_general, 4),
                "delta_guided_minus_general": round(delta, 4),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nSaved results to: {out_csv}")

    if len(out_df):
        mean_g = out_df["rougeL_guided"].mean()
        mean_gen = out_df["rougeL_general"].mean()
        print(
            f"Avg ROUGE-L — guided: {mean_g:.4f} | general: {mean_gen:.4f} | Δ = {mean_g - mean_gen:.4f}"
        )
        prop = (out_df["delta_guided_minus_general"] > 0).mean()
        print(f"Proportion guided > general: {prop:.3f}")


if __name__ == "__main__":
    main()
