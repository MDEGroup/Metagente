import pandas as pd

test_df = pd.read_csv("test_random_600-2.csv")
llama_df = pd.read_csv("LLAMA_RANDOM_600.csv")
mistral_df = pd.read_csv("MISTRAL_RANDOM_600.csv")
OG_df = pd.read_csv("test_gpt_4o_600.csv")
gemma_df = pd.read_csv("gemma_600.csv")
gpt_4o_df = pd.read_csv("results_gpt4o_600.csv")

llama_preds = llama_df["generated_summary"]
mistral_preds = mistral_df["extracted_short_desc"]
OG_preds = OG_df["Generated About"]
gemma_preds = gemma_df["pred_short_description"]
gpt_4o_preds = gpt_4o_df["output_text"]

test_df["Llama"] = llama_preds
test_df["Mistral"] = mistral_preds
test_df["OG"] = OG_preds
test_df["Gemma"] = gemma_preds
test_df["GPT-4o"] = gpt_4o_preds

# Lưu kết quả
test_df.to_csv("merged_summaries_600_by_index.csv", index=False)
