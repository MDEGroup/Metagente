EXTRACTOR_PROMPT = """
Your task is to shorten and extract only the introduction and description information from an app. You are given the following description for an app:
<Description>
$readme_text
</Description>
 
# Steps
- **Identify the structure of the app's description**: The app's description is a structure text file that might contains many sections such as introduction, description, key features, advantages, detailed use case, setup instructions, compatibility,...
- **Remove all sections that are not relevant to the app's description**: Irrelevant sections might include technical guidance (installing/running/specification... instruction), compatibility, troubleshooting,...
- **Remove all unnecessary links/tags**: Identify all links/tags that DO NOT contribute to the description of the app. You must remove all of these reference links and tags.
- **Return only text that is relevant to the description of the app**: The output should only contains the text that is relevant to the introduction/description of the app, including the app name/title, app feature description/purpose statement/overview. DO NOT include any output identifications such as: "Here's the ..." or "Extracted App's description:"
"""

INITIAL_SUMMARIZER_PROMPT = """
Summarize the following extracted text from an app's description into a short term/phrase introducing the app:
<EXTRACTED_APP'S_DESCRIPTION>
$extracted_text
</EXTRACTED_APP'S_DESCRIPTION>
 
The output should include only a short term/phrase introducing the app.
"""

TEACHER_PROMPT = """
You are a professional Prompt Engineer. You are working on a system using a Large Language Model (LLM) to help developers automatically generate a short Description term/phrase contain key concept/idea from an extracted text of the description of an app. Your task is to modify and improve the current prompt of the LLM based on the result of testing on a data include a description and a ground truth description.
 
# Steps:
- **Analyze the data for testing**: Analyze the following data include an extracted text from a description and a ground truth description from an app:
<EXTRACTED_TEXT>
$extracted_text
</EXTRACTED_TEXT>
 
<GROUND_TRUTH DESCRIPTION>
$description
</GROUND_TRUTH DESCRIPTION>
- **Review the current result**: Review the generated description using the extracted text its ROUGE score on the ground truth description to identify improvements that could be made:
<GENERATED_DESCRIPTION>
$generated_about
</GENERATED_DESCRIPTION>
<ROUGE_SCORE>
$rouge_score
</ROUGE_SCORE>
- **Prioritize extracting existing name/title, feature description/purpose statement/overview**: Compare the text from the beginning of the extracted text from description and the ground truth description. If the ground truth description is already existed in this extracted text as a name/title, feature description/purpose statement/overview, you must include in the new prompt the instruction to prioritize using it.
- **Modify the current prompt**: Identify mistakes and lacking instructions in the current prompt from the result of the above review. You should preserve the current prompt as much as possible and only make small changes to the prompt based on the identified mistakes and lacking instructions.
<CURRENT_PROMPT>
$summarizer_prompt
</CURRENT_PROMPT>
As the new prompt will not include the ground truth description, DO NOT mention about the ground truth description in the new prompt. DO NOT include any reasoning/explanation like "Based on the result of the above review:", "Here's the", ... or any output identifiers like "Prompt:", "New Prompt", ... The output should only include a string representing the new prompt for the LLM
"""

COMBINE_PROMPT = """
You are a professional Prompt Engineer. You are working on a system using a Large Language Model (LLM) to help developers automatically generate a short Description term/phrase contain key concept/idea from an extracted text of the description of an app. Your task is to combine several candidate prompts for the LLM into a final prompt.
 
# Steps:
- **Review all candidate prompts**: Analyze the following prompts to identify common parts to be included in the final prompt and also includes specific details or conditional key points from these prompts to be included in the final prompt
<CANDIDATE_PROMPTS>
$summarizer_list
</CANDIDATE_PROMPTS>
- **Generate a final prompt**: Based on the common parts and conditional key points, generate a final prompt for the LLM.

# Output Format:
Do not include any reasoning/explanation like "Based on the result of the above review:", "Here's the", ... or any output identifiers like "Prompt:", "New Prompt", ... The output should only include a string representing the prompt for the LLM
"""

ANALYSIS_PROMPT = """
A Large Language Model is used to auto-summarize app's description section and generate a concise description for an app. Your task is to analyze its result to provide a short advice on how to improve the generated description. You will be provided with the actual and the generated description section of an app and the ROUGE score between them:
<ACTUAL_DESCRIPTION>
$ground_truth
</ACTUAL_DESCRIPTION>
 
<GENERATED_DESCRIPTION>
$generated_about
</GENERATED_DESCRIPTION>
 
<ROUGE_SCORE>
$score
</ROUGE_SCORE>
 
# Steps:
- List the differences between the actual and the generated description section that results in the ROUGE score.
- Choose one main reason among the differences that best represents the ROUGE score.
- The output must be only one short advise on how to improve the generated description from the description base on that main reason.
# Output Format:
1 concise and short advice sentence
"""

ANALYSIS_SUMMARIZER_PROMPT = """
A Large Language Model is used to auto-summarize app's description section and generate a concise description for an app. You are given the following evaluating results on a dataset comparing app descriptions generated from a detail description by the Large Language Model and the actual descriptions:
<ANALYSIS_RESULT>
$analysis_result
</ANALYSIS_RESULT>
# Steps:
- Review the overall tendency of the analysis results.
- Summary one main point that best represents the analysis results.
- Give only one advise on how to improve the generated description.
# Output Format:
1 concise and short advice sentence
"""

SEQUENTIAL_TEACHER_PROMPT = """
You are a professional Prompt Engineer. You are using a Large Language Model (LLM) to help developers automatically generate a short Description term/phrase contain key concept/idea from the description of an app. The LLM include a prompt to extract the relevant information from the description and a prompt to generate a short description term/phrase contain key concept/idea. Your task is to optimize the current prompt to improve the performance of the LLM.
<CURRENT_PROMPT>
$current_summarizer_prompt
</CURRENT_PROMPT>
 
You should use the following advising analysis to help you optimize the prompts of the two agents:
<ANALYSIS_RESULT>
$current_analysis
</ANALYSIS_RESULT>
 
Here is an example of prompts that have good performance. FOLLOW this example to optimize the prompt of the LLM.
<GOOD_PROMPT>
$best_summarizer_prompt
</GOOD_PROMPT>
<SCORE>
$best_score
</SCORE>
<ADVISING_ANALYSIS>
$best_analysis_result
</ADVISING_ANALYSIS>
 
Here is an example of prompts that have bad performance. AVOID the mistakes of this example to optimize the prompt of the LLM.
<BAD_PROMPT>
$worst_summarizer_prompt
</BAD_PROMPT>
<SCORE>
$worst_score
</SCORE>
<ADVISING_ANALYSIS>
$worst_analysis_result
</ADVISING_ANALYSIS>
 
You should preserve the current prompt as much as possible and only make small changes to the prompt based on the advising analysis. You must include in the detail part that the description is "A shortest term or phrase include only the concept/idea of the app, without any explanations or details". The answer must only include the new prompt for the LLM
 
# Output Format:
Prompt: <Prompt>
"""

ANALYZER_PROMPT = """
You are an Analyzer Agent. Your task is to analyze the given document and extract structured information
needed to build prompts for downstream agents.

<Document>
$document
</Document>

# Steps:
1. Identify what type of domain this document belongs to (e.g., app, readme, paper, landing page...).
2. Define the most appropriate "data_type" (e.g., text, paragraph, abstract, section).
3. List the important "features" sections to keep (intro, overview, key features, value proposition...).
4. List the "irrelevant_features" sections to discard (installation, troubleshooting, references...).
5. List the "unnecessary_feature" content formats to remove (links, tags, badges...).
6. List the must-have "feature" fields that should always appear (app name, purpose, 1-sentence value prop...).

# Output Format:
Return JSON with the following keys:
{
  "domain": "<string>",
  "data_type": "<string>",
  "features": ["..."],
  "irrelevant_features": ["..."],
  "unnecessary_feature": ["..."],
  "feature": ["..."]
}
"""

OPTIMIZED_SUMMARIZER_PROMPT = """"Summarize the following extracted text from an app's description into a short term/phrase introducing the app: 

<EXTRACTED_APP'S_DESCRIPTION> 
$extracted_text 
</EXTRACTED_APP'S_DESCRIPTION> 

The output should include only a short term/phrase introducing the app."""
