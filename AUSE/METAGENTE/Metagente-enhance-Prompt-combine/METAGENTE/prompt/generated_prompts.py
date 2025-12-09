# Auto-generated prompts

EXTRACTOR_PROMPT = """Your task is to extract only the core introduction and description suitable for product_page.

Rules:
- Remove any forbidden elements.
- Prioritize in order: app name, features, target audience.
- Target length: 100 words.
- Focus on clarity.

Input:
<Doc>
$readme_text
</Doc>"""

INITIAL_SUMMARIZER_PROMPT = """Summarize the extracted description into a short introduction for the product_page, focusing on the app name, features, and target audience while ensuring clarity. Exclude any forbidden elements.

<EXTRACTED>
$extracted_text 
</EXTRACTED>"""

TEACHER_PROMPT = """You are a Prompt Engineer improving a summarizer prompt for a product page aimed at knowledge workers, students, and individuals seeking to enhance focus. Focus on enhancing clarity while ensuring the summary includes the app name, features, and target audience.

Data:
<EXTRACTED_TEXT>
$extracted_text
</EXTRACTED_TEXT>
<GROUND_TRUTH>
$description
</GROUND_TRUTH>
<GENERATED>
$generated_about
</GENERATED>
<ROUGE_L>
$rouge_score
</ROUGE_L>

Constraints:
- Preserve structure; edit minimally.
- Enforce exclusions: Here's the ..., Extracted App's description:, The summary is:, In summary,, Overall,, This document describes ..., Download from [focusflow.app](https://focusflow.app)., Install on your device., Create an account to sync across devices.
- Keep priority order: app name, features, target audience
- Do not mention ground truth in the new prompt.

<CURRENT_PROMPT>
$summarizer_prompt
</CURRENT_PROMPT>"""

COMBINE_PROMPT = """You combine multiple candidate summarizer prompts into one final prompt for product_page. Emphasize clarity while focusing on the app name, features, and target audience of knowledge workers, students, and individuals seeking to improve focus. Ensure the summary adheres to a length policy of 100 and avoids forbidden elements.

<CANDIDATE_PROMPTS>
$summarizer_list 
</CANDIDATE_PROMPTS>"""
