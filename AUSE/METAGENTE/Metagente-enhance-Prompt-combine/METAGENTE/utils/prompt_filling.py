from string import Template

CORE_SLOTS = [
    "domain",
    "data_type",
    "features",
    "irrelevant_features",
    "unnecessary_feature",
    "feature",
]


def prepare_slots(analyzed: dict, **overrides) -> dict:
    """
    Hợp nhất 6 trường Analyzer với các biến bổ sung cho từng agent (vd: readme_text, extracted_text...).
    - analyzed: output của AnalyzerAgent.run(document)
    - overrides: các biến chuyên biệt theo lượt gọi agent
    """
    slots = {k: analyzed.get(k, "") for k in CORE_SLOTS}
    # slots.update(overrides or {})

    for k, v in list(slots.items()):
        slots[k.upper()] = str(v).upper()
    return slots


def fill_prompt(prompt_template: str, slots: dict) -> str:
    return Template(prompt_template).safe_substitute(**slots)
