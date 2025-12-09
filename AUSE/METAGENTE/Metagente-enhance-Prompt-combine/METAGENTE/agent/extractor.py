from string import Template

from utils.llm import OpenAIClient


class ExtractorAgent:
    def __init__(self):
        self.llm = OpenAIClient(model="gpt-4o-mini")

    def _build_prompt(self, prompt: str, readme_text: str) -> str:
        prompt = Template(prompt)
        prompt = prompt.substitute(readme_text=readme_text)
        return prompt

    def run(self, prompt: str, readme_text: str) -> str:
        prompt = self._build_prompt(prompt, readme_text)
        extracted_text = self.llm.generate(prompt, temperature=0)
        return extracted_text
