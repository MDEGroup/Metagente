import re
from string import Template

from prompt.prompt import SEQUENTIAL_TEACHER_PROMPT
from utils.llm import OpenAIClient


class SequentialTeacherAgent:
    def __init__(self):
        self.llm = OpenAIClient(model="gpt-3.5-turbo")

    def _format(self, analysis_result: list) -> str:
        clean_result = ""
        for i, result in enumerate(analysis_result):
            clean_result += f"""Prompts #{i}:\nExtractor Prompt: {result["extractor_prompt"]}\nSummarizer Prompt: {result["summarizer_prompt"]}\nScore: {result["average_rouge_score"]}\nAdvising Analysis: {result["analysis_summary"]}\n"""
        return clean_result

    def _build_prompt(
        self,
        best_summarizer_prompt: str,
        best_analysis_result: str,
        best_score: float,
        worst_summarizer_prompt: str,
        worst_analysis_result: str,
        worst_score: float,
        current_summarizer_prompt: str,
        current_analysis_result: str,
    ) -> str:
        prompt = Template(SEQUENTIAL_TEACHER_PROMPT)
        prompt = prompt.substitute(
            best_summarizer_prompt=best_summarizer_prompt,
            best_analysis_result=best_analysis_result,
            best_score=best_score,
            worst_summarizer_prompt=worst_summarizer_prompt,
            worst_analysis_result=worst_analysis_result,
            worst_score=worst_score,
            current_summarizer_prompt=current_summarizer_prompt,
            current_analysis=current_analysis_result,
        )
        return prompt

    def _parse_answer(self, answer: str) -> tuple[str, str]:
        summarizer_pattern = r"Prompt:\s*(.*)"
        try:
            summarizer_prompt = (
                re.search(summarizer_pattern, answer, re.DOTALL).group(1).strip()
            )
        except Exception:
            summarizer_prompt = (
                re.search(r"Prompt\s*(.*)", answer, re.DOTALL).group(1).strip()
            )
        if "$extracted_text" not in summarizer_prompt:
            summarizer_prompt = re.sub(
                r"\btext\b", "$extracted_text", summarizer_prompt, count=1
            )

        return summarizer_prompt

    def run(
        self,
        best_summarizer_prompt: str,
        best_analysis_summary: str,
        best_score: float,
        worst_summarizer_prompt: str,
        worst_analysis_summary: str,
        worst_score: float,
        current_summarizer_prompt: str,
        current_analysis_summary: str,
    ) -> tuple[str, str]:
        prompt = self._build_prompt(
            best_summarizer_prompt,
            best_analysis_summary,
            best_score,
            worst_summarizer_prompt,
            worst_analysis_summary,
            worst_score,
            current_summarizer_prompt,
            current_analysis_summary,
        )
        answer = self.llm.generate(prompt, temperature=0.7)

        print(f"Teacher raw answer: {answer}")

        summarizer_prompt = self._parse_answer(answer)
        return summarizer_prompt
