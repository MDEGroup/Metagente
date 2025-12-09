import re
from string import Template

from prompt.prompt import TEACHER_PROMPT
from utils.llm import OpenAIClient


class TeacherAgent:
    def __init__(self, prompt: str = TEACHER_PROMPT):
        self.llm = OpenAIClient(model="gpt-4o")
        self.prompt = prompt

    def _build_prompt(
        self,
        prompt: str,
        extracted_text: str,
        description: str,
        generated_about: str,
        rouge_score: float,
        summarizer_prompt: str,
    ) -> str:
        prompt_up = Template(prompt)
        prompt_up = prompt_up.substitute(
            extracted_text=extracted_text,
            description=description,
            generated_about=generated_about,
            rouge_score=rouge_score,
            summarizer_prompt=summarizer_prompt,
        )
        return prompt_up

    def _parse_answer(self, answer: str) -> tuple[str, str]:
        summarizer_prompt = answer
        if "$extracted_text" not in summarizer_prompt:
            summarizer_prompt = re.sub(
                r"\btext\b", "$extracted_text", summarizer_prompt, count=1
            )

        return summarizer_prompt

    def run(
        self,
        prompt: str,
        extracted_text: str,
        description: str,
        generated_about: str,
        rouge_score: float,
        summarizer_prompt: str,
    ) -> tuple[str, str]:
        prompt_up = self._build_prompt(
            prompt,
            extracted_text,
            description,
            generated_about,
            rouge_score,
            summarizer_prompt,
        )
        print(f"Teacher prompt: {prompt_up}")
        answer = self.llm.generate(prompt_up, temperature=0.7)

        print(f"Teacher raw answer: {answer}")

        summarizer_prompt = self._parse_answer(answer)
        return summarizer_prompt
