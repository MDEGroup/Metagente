EXTRACTOR_PROMPT = """
Your task is to shorten and extract only the introduction and description information from a $domain. You are given the following description for a $domain:
<Description>
$readme_text
</Description>
 
# Steps
- **Identify the structure of the $domain**: The $domain is a structure $data_type that might contains many sections such as $features,...
- **Remove all sections that are not relevant to the $domain**: Irrelevant sections might include $irrelevant_features.
- **Remove all unnecessary $unnecessary_feature**: Identify all $unnecessary_feature that DO NOT contribute to the $domain. You must remove all of these reference $unnecessary_feature.
- **Return only $data_type that is relevant to the $domain**: The output should only contains the $data_type that is relevant to the introduction/description of the $domain, including the $feature. DO NOT include any output identifications such as: "Here's the ..." or "Extracted $domain:"
"""
INITIAL_SUMMARIZER_PROMPT = """
Summarize the following extracted text from an $domain into a short term/phrase introducing the $domain:
<EXTRACTED_$DOMAIN>
$extracted_text
</EXTRACTED_$DOMAIN>
 
The output should include only a short term/phrase introducing the $domain.
"""

TEACHER_PROMPT = """
You are a professional Prompt Engineer. You are working on a system using a Large Language Model (LLM) to help developers automatically generate a short Description term/phrase contain key concept/idea from an extracted $data_type of the $domain. Your task is to modify and improve the current prompt of the LLM based on the result of testing on a data include a description and a ground truth description.
 
# Steps:
- **Analyze the data for testing**: Analyze the following data include an extracted $data_type from a description and a ground truth description from $domain:
<EXTRACTED_TEXT>
$extracted_text
</EXTRACTED_TEXT>
 
<GROUND_TRUTH DESCRIPTION>
$description
</GROUND_TRUTH DESCRIPTION>
- **Review the current result**: Review the generated description using the extracted $data_type its ROUGE score on the ground truth description to identify improvements that could be made:
<GENERATED_DESCRIPTION>
$generated_about
</GENERATED_DESCRIPTION>
<ROUGE_SCORE>
$rouge_score
</ROUGE_SCORE>
- **Prioritize extracting existing $features**: Compare the text from the beginning of the extracted $data_type from description and the ground truth description. If the ground truth description is already existed in this extracted $data_type as a $features, you must include in the new prompt the instruction to prioritize using it.
- **Modify the current prompt**: Identify mistakes and lacking instructions in the current prompt from the result of the above review. You should preserve the current prompt as much as possible and only make small changes to the prompt based on the identified mistakes and lacking instructions.
<CURRENT_PROMPT>
$summarizer_prompt
</CURRENT_PROMPT>
As the new prompt will not include the ground truth description, DO NOT mention about the ground truth description in the new prompt. DO NOT include any reasoning/explanation like "Based on the result of the above review:", "Here's the", ... or any output identifiers like "Prompt:", "New Prompt", ... The output should only include a string representing the new prompt for the LLM
"""

COMBINE_PROMPT = """
You are a professional Prompt Engineer. You are working on a system using a Large Language Model (LLM) to help developers automatically generate a short Description term/phrase contain key concept/idea from an extracted description of the description. Your task is to combine several candidate prompts for the LLM into a final prompt.
 
# Steps:
- **Review all candidate prompts**: Analyze the following prompts to identify common parts to be included in the final prompt and also includes specific details or conditional key points from these prompts to be included in the final prompt
<CANDIDATE_PROMPTS>
$summarizer_list
</CANDIDATE_PROMPTS>
- **Generate a final prompt**: Based on the common parts and conditional key points, generate a final prompt for the LLM.

# Output Format:
Do not include any reasoning/explanation like "Based on the result of the above review:", "Here's the", ... or any output identifiers like "Prompt:", "New Prompt", ... The output should only include a string representing the prompt for the LLM
"""
