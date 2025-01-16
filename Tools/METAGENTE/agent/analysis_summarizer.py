from string import Template

from prompt.prompt import ANALYSIS_SUMMARIZER_PROMPT
from utils.llm import AzureOpenAIClient


class AnalysisSummarizerAgent:
    def __init__(self):
        self.llm = AzureOpenAIClient(model="gpt-4o-mini")

    def _format_analysis_result(self, analysis_result: list) -> str:
        structure_result = ""
        for i, result in enumerate(analysis_result):
            structure_result += f"""Data #{i}:\nActual Description: {result["description"]}\nGenerated Description: {result["generated_about"]}\nAnalysis Result: {result["analysis_reasoning"]}\n"""
        return structure_result

    def _build_prompt(self, analysis_result: list) -> str:
        prompt = Template(ANALYSIS_SUMMARIZER_PROMPT)
        prompt = prompt.substitute(analysis_result=analysis_result)
        return prompt

    def run(self, analysis_result: list) -> str:
        prompt = self._build_prompt(self._format_analysis_result(analysis_result))
        analysis_result = self.llm.generate(prompt)
        return analysis_result
