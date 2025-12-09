# Auto-generated prompts

EXTRACTOR_PROMPT = """Your task is to extract only the core introduction and description suitable for product_page .

FocusFlow is designed for knowledge workers, students, and individuals seeking enhanced focus. This app offers features such as task management, time tracking, and customizable workflows to help users optimize their productivity. With an intuitive interface, FocusFlow enables users to streamline their tasks and maintain concentration, making it an essential tool for anyone looking to improve their efficiency.

Input:
<Doc>
$readme_text
</Doc>

Output: plain text only, no labels."""

INITIAL_SUMMARIZER_PROMPT = """FocusFlow is an app designed for knowledge workers, students, and individuals seeking enhanced focus. It offers features that help users manage their time effectively, minimize distractions, and boost productivity.

<EXTRACTED>
$extracted_text
</EXTRACTED>

Output only the summary string."""

TEACHER_PROMPT = """Revise the summarizer prompt for a product page aimed at knowledge workers, students, and individuals seeking focus. Focus on enhancing clarity through minimal yet impactful edits. Ensure the summary excludes specific phrases and maintains the priority of app name, features, and target audience.

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

Output only the new prompt string."""

COMBINE_PROMPT = """Combine candidate prompts into a final product page summary. Focus on clarity for knowledge workers, students, and individuals seeking focus. Highlight the app name, key features, and target audience while avoiding forbidden phrases. Ensure the summary is concise and adheres to a length of 100 words.

<CANDIDATE_PROMPTS>
$summarizer_list
</CANDIDATE_PROMPTS>

Output only the final prompt string."""

