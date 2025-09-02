from string import Template

from prompt.prompt import ANALYSIS_PROMPT
from utils.llm import OpenAIClient


class AnalysisAgent:
    def __init__(self):
        self.llm = OpenAIClient(model="gpt-3.5-turbo")

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
