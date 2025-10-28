import pandas as pd
from rouge_score import rouge_scorer

# Read the merged file by index
df = pd.read_csv("merged_summaries_600_by_index.csv")

# Create ROUGE scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


# Function to compute ROUGE for each row
def compute_rouge_scores(reference, prediction):
    if pd.isna(prediction) or pd.isna(reference):
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


# List of models
models = ["Gitsum", "Llama", "Mistral", "GPT-4o", "OG", "Gemma"]

# Compute ROUGE for each model
for model in models:
    scores = df.apply(
        lambda row: compute_rouge_scores(row["description_short"], row[model]), axis=1
    )
    # Assign results to new columns
    df[f"{model}_ROUGE-1"] = [float(s["rouge1"]) for s in scores]
    df[f"{model}_ROUGE-2"] = [float(s["rouge2"]) for s in scores]
    df[f"{model}_ROUGE-L"] = [float(s["rougeL"]) for s in scores]

# Explicitly cast ROUGE columns to float
for model in models:
    for rouge_type in ["ROUGE-1", "ROUGE-2", "ROUGE-L"]:
        col = f"{model}_{rouge_type}"
        df[col] = df[col].astype(float)

# Save the output file with float formatting
output_path = "score/merged_summaries_600_with_html_and_rouge.csv"
df.to_csv(output_path, index=False, float_format="%.6f")

print(f"Results with ROUGE scores have been saved to: {output_path}")
