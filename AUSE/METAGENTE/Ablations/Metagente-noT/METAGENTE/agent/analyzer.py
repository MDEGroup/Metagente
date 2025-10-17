import json
import re
from string import Template
from typing import Any, Dict

from prompt.prompt import ANALYZER_PROMPT
from utils.llm import OpenAIClient


class AnalyzerAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = OpenAIClient(model=model)

    def _build_prompt(self, document: str) -> str:
        return Template(ANALYZER_PROMPT).substitute(document=document)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        s = text.strip()

        # bóc ```json ... ```
        fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", s)
        if fence:
            s = fence.group(1).strip()

        # thử parse trực tiếp
        try:
            return json.loads(s)
        except Exception:
            pass

        # fallback: bắt {...}
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            candidate = m.group(0)
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            return json.loads(candidate)

        raise ValueError("Analyzer must return valid JSON. Got:\n" + text[:500])

    def run(self, document: str) -> Dict[str, Any]:
        prompt = self._build_prompt(document)
        raw = self.llm.generate(prompt, temperature=0)
        data = self._extract_json(raw)

        # chuẩn hoá đảm bảo đủ 6 trường
        fields = [
            "domain",
            "data_type",
            "features",
            "irrelevant_features",
            "unnecessary_feature",
            "feature",
        ]
        for f in fields:
            data.setdefault(f, "" if f in ("domain", "data_type") else [])
        return data
