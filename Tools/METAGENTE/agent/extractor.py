from utils.llm import AzureOpenAIClient
from string import Template


class ExtractorAgent:
    def __init__(self):
        self.llm = AzureOpenAIClient(model="gpt-4o-mini")

    def _build_prompt(self, prompt: str, readme_text: str) -> str:
        prompt = Template(prompt)
        prompt = prompt.substitute(readme_text=readme_text)
        return prompt

    def run(self, prompt: str, readme_text: str) -> str:
        prompt = self._build_prompt(prompt, readme_text)
        extracted_text = self.llm.generate(prompt, temperature=0)
        return extracted_text
