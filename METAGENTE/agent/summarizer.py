from string import Template

from utils.llm import AzureOpenAIClient


class SummarizerAgent:
    def __init__(self):
        self.llm = AzureOpenAIClient(model="gpt-4o-mini")

    def _build_prompt(self, prompt: str, extracted_text: str) -> str:
        prompt = Template(prompt)
        prompt = prompt.substitute(extracted_text=extracted_text)
        return prompt

    def run(self, prompt: str, extracted_text: str) -> str:
        prompt = self._build_prompt(prompt, extracted_text)
        about = self.llm.generate(prompt, temperature=0)
        return about
