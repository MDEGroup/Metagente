from __future__ import annotations

import json
import re
from dataclasses import dataclass
from string import Template
from typing import Any, Dict, List, Tuple

# --------------------------------
# Utilities
# --------------------------------
from prompt.prompt_template import REGION_ADAPTER_PROMPT
from utils.llm import OpenAIClient


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [str(x)]


def _join(items: List[Any]) -> str:
    return ", ".join(map(str, items))


def _find_placeholders(s: str) -> List[str]:
    """Find $placeholders used by string.Template (names only, without the $)."""
    return sorted(set(m.group(1) for m in re.finditer(r"\$(?:\{)?([a-zA-Z_][\w]*)", s)))


# ---------- Region helpers (hard safety) ----------

REFINE_OPEN = r"\[\[REFINE\]\]"
REFINE_CLOSE = r"\[\[/REFINE\]\]"


def _split_refine_regions(text: str) -> List[Tuple[str, str, str]]:
    """
    Split text into a list of (prefix, region, suffix) segments for each [[REFINE]]...[[/REFINE]] pair.
    If multiple regions exist, returns them sequentially for reconstruction.

    Returns list of triples:
      - before: text before region
      - region: inner content to be refined
      - after: remaining text (which may contain more refine pairs)
    """
    parts: List[Tuple[str, str, str]] = []
    cur = text
    pattern = re.compile(f"{REFINE_OPEN}(.*?){REFINE_CLOSE}", flags=re.DOTALL)
    pos = 0
    while True:
        m = pattern.search(cur, pos)
        if not m:
            break
        start, end = m.span()
        before = cur[:start]
        region = m.group(1)
        after = cur[end:]
        parts.append((before, region, after))
        # Prepare for next search in the remainder
        cur = after
        pos = 0
    # Nếu không có vùng nào, trả về danh sách rỗng
    return parts


def _reconstruct_from_regions(original: str, refined_regions: List[str]) -> str:
    """
    Dựa trên original text có n vùng REFINE, ghép từng vùng đã refine (refined_regions[i]) vào đúng vị trí.
    """
    out = []
    cur = original
    pattern = re.compile(f"{REFINE_OPEN}(.*?){REFINE_CLOSE}", flags=re.DOTALL)
    idx = 0
    while True:
        m = pattern.search(cur)
        if not m:
            out.append(cur)
            break
        start, end = m.span()
        out.append(cur[:start])
        out.append(refined_regions[idx])
        idx += 1
        cur = cur[end:]
    return "".join(out)


# --------------------------------
# DomainView
# --------------------------------


@dataclass
class DomainView:
    language: str
    target_audience: str
    reading_level: str
    doc_type: str
    forbidden_elements: List[str]
    keywords_priority: List[str]
    summary_length: str
    scoring_focus: List[str]

    @classmethod
    def from_analyzer(cls, analyzer_json: Dict[str, Any]) -> "DomainView":
        dj = analyzer_json or {}
        domain = dj.get("domain_traits", {})
        style = domain.get("style_constraints", {})
        return cls(
            language=dj.get("language", "en"),
            target_audience=dj.get("target_audience", "end users"),
            reading_level=dj.get("reading_level", "layperson"),
            doc_type=dj.get("doc_type", "app_readme"),
            forbidden_elements=_as_list(
                style.get("forbidden_elements", ["links", "install", "badges"])
            ),
            # dùng key_entities (generic) làm ưu tiên keywords
            keywords_priority=_as_list(
                style.get("key_entities", ["name/title", "purpose", "unique_value"])
            ),
            summary_length=str(style.get("summary_length", "one_sentence")),
            scoring_focus=_as_list(domain.get("scoring_focus", ["clarity"])),
        )

    def to_subs(self) -> Dict[str, str]:
        return {
            "language": self.language,
            "target_audience": self.target_audience,
            "reading_level": self.reading_level,
            "doc_type": self.doc_type,
            "forbidden_elements": _join(self.forbidden_elements),
            "keywords_priority": _join(self.keywords_priority),
            "summary_length": self.summary_length,
            "scoring_focus": _join(self.scoring_focus),
        }


# --------------------------------
# Role names (with short descriptions)
# --------------------------------

ROLE_NAME_EXTRACTOR = (
    "EXTRACTOR - Filters and extracts only relevant introduction/description text"
)
ROLE_NAME_SUMMARIZER = (
    "SUMMARIZER - Condenses extracted text into a short summary phrase"
)
ROLE_NAME_TEACHER = "TEACHER - Reviews summarizer output, ROUGE score, and adjusts prompt to improve quality"
ROLE_NAME_COMBINER = (
    "COMBINER - Combines multiple candidate prompts into a final optimized prompt"
)

# --------------------------------
# Region-only refiner prompt
# --------------------------------


# --------------------------------
# Base Prompt Creator (region-only refinement)
# --------------------------------


class BasePromptCreator:
    """
    Region-only refinement:
    - Fill domain placeholders.
    - Detect [[REFINE]]...[[/REFINE]] regions.
    - For each region: send ONLY its inner text to LLM with REGION_ADAPTER_PROMPT, get rewritten text back.
    - Reconstruct the full prompt by replacing the regions; everything else (including $placeholders and <BLOCKS>) stays untouched.

    => Placeholder gốc và các block XML-like tuyệt đối không bị chạm tới.
    """

    ROLE_NAME: str = "GENERIC"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        use_refiner: bool = True,
        temperature: float = 0.2,
    ):
        self.llm = OpenAIClient(model=model)
        self.use_refiner = use_refiner
        self.temperature = temperature

    def build(self, template_str: str, analyzer_json: Dict[str, Any]) -> str:
        # 0) Chụp lại placeholder ban đầu (để tự tin rằng ta không làm gì động vào chúng)
        original_placeholders = set(_find_placeholders(template_str))

        # 1) Điền domain placeholders
        view = DomainView.from_analyzer(analyzer_json)
        subs = view.to_subs()
        filled = Template(template_str).safe_substitute(**subs)

        if not self.use_refiner:
            # đảm bảo placeholder runtime vẫn y nguyên (vì ta chỉ thay domain keys)
            self._assert_runtime_placeholders_intact(
                filled, original_placeholders, subs.keys()
            )
            return filled

        # 2) Tìm các vùng [[REFINE]]...[[/REFINE]]
        regions = _split_refine_regions(filled)
        if not regions:
            # Không có vùng cho phép chỉnh -> không refine
            self._assert_runtime_placeholders_intact(
                filled, original_placeholders, subs.keys()
            )
            return filled

        # 3) Refine từng vùng, giữ nguyên mọi phần còn lại
        refined_chunks: List[str] = []
        cur_text = filled
        for idx, (before, region, after) in enumerate(regions, start=1):
            refined_region = self._refine_region(region, view, role=self.ROLE_NAME)
            refined_chunks.append(refined_region)

        # 4) Ghép lại đầy đủ
        final_text = _reconstruct_from_regions(filled, refined_chunks)

        # 5) Placeholder & blocks không bị chạm tới (vì không gửi cho LLM). Kiểm tra lại để chắc chắn.
        self._assert_runtime_placeholders_intact(
            final_text, original_placeholders, subs.keys()
        )
        self._ensure_blocks_unchanged(filled, final_text)

        return final_text

    def _refine_region(self, region_text: str, view: DomainView, *, role: str) -> str:
        meta = Template(REGION_ADAPTER_PROMPT).substitute(
            role=role,
            language=view.language,
            target_audience=view.target_audience,
            reading_level=view.reading_level,
            doc_type=view.doc_type,
            forbidden_elements=", ".join(view.forbidden_elements),
            keywords_priority=", ".join(view.keywords_priority),
            summary_length=view.summary_length,
            scoring_focus=", ".join(view.scoring_focus),
            domain_view_json=json.dumps(view.__dict__, ensure_ascii=False),
            region=region_text,
        )
        out = self.llm.generate(meta, temperature=self.temperature).strip()
        # loại bỏ code fences nếu có
        if out.startswith("```"):
            out = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", out, flags=re.MULTILINE).strip()
        # tuyệt đối không cho placeholder lọt vào vùng
        if _find_placeholders(out):
            raise ValueError("Refined region must not introduce placeholders.")
        return out

    @staticmethod
    def _assert_runtime_placeholders_intact(
        text: str, original_ph: set, domain_keys: set
    ) -> None:
        now = set(_find_placeholders(text))
        expected_runtime = original_ph - set(domain_keys)
        missing = [p for p in expected_runtime if p not in now]
        if missing:
            raise ValueError(
                f"Runtime placeholders were modified or removed: {missing}"
            )

    @staticmethod
    def _ensure_blocks_unchanged(before: str, after: str) -> None:
        """
        Đảm bảo các block XML-like (nếu có) vẫn giữ nguyên. Ở đây ta không mask, nhưng vì không refine ngoài vùng,
        chúng không bị chạm tới — check đơn giản bằng sự hiện diện tương ứng.
        """
        block_names = re.findall(r"<([A-Z_]+)>", before)  # e.g., Doc/EXTRACTED/...
        for name in set(block_names):
            if f"<{name}>" in before and f"</{name}>" in before:
                if f"<{name}>" not in after or f"</{name}>" not in after:
                    raise ValueError(f"Block <{name}> was altered unexpectedly")


# --------------------------------
# Roles
# --------------------------------


class ExtractorPromptCreator(BasePromptCreator):
    ROLE_NAME = ROLE_NAME_EXTRACTOR


class SummarizerPromptCreator(BasePromptCreator):
    ROLE_NAME = ROLE_NAME_SUMMARIZER


class TeacherPromptCreator(BasePromptCreator):
    ROLE_NAME = ROLE_NAME_TEACHER


class CombinerPromptCreator(BasePromptCreator):
    ROLE_NAME = ROLE_NAME_COMBINER
