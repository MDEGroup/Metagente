OPTIMIZED_SUMMARIZER_PROMPT = """"Summarize the following extracted text from an app's description into a short term/phrase introducing the app. Prioritize using any existing name/title, feature description, or purpose statement found at the beginning of the extracted text. Ensure the output accurately reflects the app's primary function, target audience, and any specific cultural or regional focus mentioned. The output should include only a short term/phrase introducing the app.

<EXTRACTED_APP'S_DESCRIPTION>
$extracted_text
</EXTRACTED_APP'S_DESCRIPTION>"""
