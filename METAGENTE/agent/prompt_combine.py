import re
from string import Template

from prompt.prompt import COMBINE_PROMPT
from utils.llm import AzureOpenAIClient


class PromptCombineAgent:
    def __init__(self):
        self.llm = AzureOpenAIClient(model="gpt-4o")

    def _build_prompt(self, summarizer_list: str) -> str:
        prompt = Template(COMBINE_PROMPT)
        prompt = prompt.substitute(summarizer_list=summarizer_list)
        return prompt

    def _parse_answer(self, answer: str) -> tuple[str, str]:
        summarizer_prompt = answer
        if "$extracted_text" not in summarizer_prompt:
            summarizer_prompt = re.sub(
                r"\btext\b", "$extracted_text", summarizer_prompt, count=1
            )

        return summarizer_prompt

    def _clean_prompt_list(self, prompt_list: list) -> list:
        ret = ""
        for i, prompt in enumerate(prompt_list):
            ret += f"Extractor Prompts #{i}:\n{prompt}"
        return ret

    def run(self, prompt_list: list) -> tuple[str, str]:
        summarizer_list = self._clean_prompt_list(prompt_list)
        prompt = self._build_prompt(summarizer_list=summarizer_list)
        answer = self.llm.generate(prompt, temperature=0.2)

        print(f"Teacher raw answer: {answer}")

        summarizer_prompt = self._parse_answer(answer)
        return summarizer_prompt
