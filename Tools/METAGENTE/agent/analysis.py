from string import Template

from prompt.prompt import ANALYSIS_PROMPT
from utils.llm import AzureOpenAIClient


class AnalysisAgent:
    def __init__(self):
        self.llm = AzureOpenAIClient(model="gpt-4o-mini")

    def _build_prompt(
        self, generated_about: str, ground_truth: str, score: float
    ) -> str:
        prompt = Template(ANALYSIS_PROMPT)
        prompt = prompt.substitute(
            generated_about=generated_about, ground_truth=ground_truth, score=score
        )
        return prompt

    def run(self, generated_about: str, ground_truth: str, score: float) -> str:
        prompt = self._build_prompt(generated_about, ground_truth, score)
        analysis_result = self.llm.generate(prompt)
        return analysis_result
