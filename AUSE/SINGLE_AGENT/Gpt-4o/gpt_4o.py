import re

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ==== CONFIG ====
INPUT_FILE = "test_random_600-2.csv"
OUTPUT_FILE = "results_gpt4o_600.csv"
MODEL = "gpt-4o"
HTML_COL = "description_html_clean"
MAX_ROWS = 0


def get_prompt(description_html):
    return (
        "You are an expert app store editor. "
        "Given the following app description in HTML format, summarize it in 2-3 sentences, "
        "with a concise, engaging short description (max 80 characters) suitable for an app store listing. "
        f"App Description HTML:\n{description_html}\n"
        "Format your response as:\n"
        "Short Description: <your short description>\n\n"
    )


def clean_output(text: str) -> str:
    """
    Loại bỏ tiền tố 'Short Description:' nếu có,
    và strip khoảng trắng thừa.
    """
    if not text:
        return ""

    m = re.search(r"Short Description:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def main():
    api_key = "your_openai_api_key"
    if not api_key:
        raise RuntimeError("set OPENAI_API_KEY before running")

    df = pd.read_csv(INPUT_FILE)
    if MAX_ROWS and MAX_ROWS > 0:
        df = df.head(MAX_ROWS)

    client = OpenAI(api_key=api_key)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        html = str(row[HTML_COL]) if pd.notna(row[HTML_COL]) else ""
        prompt = get_prompt(html)

        try:
            resp = client.responses.create(model=MODEL, input=prompt)
            text_raw = resp.output_text or ""
        except Exception as e:
            text_raw = f"[ERROR] {e}"

        text_clean = clean_output(text_raw)

        results.append(
            {
                "description_html_clean": row.get("description_html_clean"),
                "description_short": row.get("description_short"),
                "output_text": text_clean,
            }
        )

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Done! Save as {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
