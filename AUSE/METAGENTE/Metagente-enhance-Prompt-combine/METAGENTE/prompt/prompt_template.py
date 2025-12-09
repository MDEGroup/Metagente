EXTRACTOR_TEMPLATE_EXAMPLE = """
Your task is to extract only the core introduction and description suitable for $doc_type .

[[REFINE]]
Integrate these constraints concisely (no meta/filler words):
- Keep language: $language
- Strictly remove: $forbidden_elements
- Prioritize in order: $keywords_priority
- Target length policy: $summary_length
[[/REFINE]]

Input:
<Doc>
$readme_text
</Doc>

Output: plain text only, no labels.
"""


SUMMARIZER_TEMPLATE_EXAMPLE = """
[[REFINE]]
Rewrite the instruction to produce a single, tight summary introducing the $doc_type.
- Prioritize: $keywords_priority
- Exclude: $forbidden_elements
- Be direct and concrete; no meta/filler phrases.
[[/REFINE]]

<EXTRACTED>
$extracted_text
</EXTRACTED>

Output only the summary string.
"""


TEACHER_TEMPLATE_EXAMPLE = """
[[REFINE]]
You are a Prompt Engineer improving a summarizer prompt for $doc_type targeted at $target_audience ($reading_level).
Emphasize improvements aligned with: $scoring_focus.
Keep edits minimal but effective; prefer surgical changes over full rewrites.

Constraints to enforce in the revised prompt:
- Enforce exclusions: $forbidden_elements
- Keep priority order: $keywords_priority
- Do not mention the ground truth explicitly.
[[/REFINE]]

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

<CURRENT_PROMPT>
$summarizer_prompt
</CURRENT_PROMPT>

Output only the new prompt string.
"""


COMBINER_TEMPLATE_EXAMPLE = """
[[REFINE]]
Combine multiple candidate summarizer prompts into one final prompt for $doc_type.
- Audience: $target_audience ($reading_level)
- Emphasize: $scoring_focus
- Forbid: $forbidden_elements
- Respect priority: $keywords_priority
- Honor length policy: $summary_length
Be succinct; keep shared core rules and fold in conditional specifics where useful.
[[/REFINE]]

<CANDIDATE_PROMPTS>
$summarizer_list
</CANDIDATE_PROMPTS>

Output only the final prompt string.
""".strip()

REGION_ADAPTER_PROMPT = """
You are a prompt engineer for ROLE: ${role}.

You will receive:
- A DOMAIN VIEW (JSON)
- A REGION_TEXT extracted from a larger prompt template.
Your job is to rewrite REGION_TEXT to best serve the ROLE and the domain constraints.

STRICT RULES:
- Output ONLY the rewritten REGION_TEXT content. Do NOT add any extra prefixes/suffixes.
- Do NOT introduce or reference any variables/placeholders (e.g., $$name).
- Keep the meaning and constraints, but you may shorten, reorder, and clarify.
- Do NOT mention these rules in your output.

Domain constraints to honor (may be integrated implicitly):
- Language: ${language}
- Audience: ${target_audience} (${reading_level})
- Doc type: ${doc_type}
- Forbidden elements: ${forbidden_elements}
- Priority keywords: ${keywords_priority}
- Length policy: ${summary_length}
- Scoring focus: ${scoring_focus}

DOMAIN VIEW (JSON):
${domain_view_json}

REGION_TEXT:

${region}

""".strip()
