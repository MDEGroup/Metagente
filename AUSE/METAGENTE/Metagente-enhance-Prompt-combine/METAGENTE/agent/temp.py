from pathlib import Path

from agent.analyzer import AnalyzerAgent
from agent.prompt_creator import (
    CombinerPromptCreator,
    ExtractorPromptCreator,
    SummarizerPromptCreator,
    TeacherPromptCreator,
)
from prompt.prompt import ANALYZER_PROMPT
from prompt.prompt_template import (
    COMBINER_TEMPLATE_EXAMPLE,
    EXTRACTOR_TEMPLATE_EXAMPLE,
    SUMMARIZER_TEMPLATE_EXAMPLE,
    TEACHER_TEMPLATE_EXAMPLE,
)

doc = """# FocusFlow

FocusFlow helps you block distractions and maintain deep work sessions.  
It is designed for knowledge workers, students, and anyone who struggles with staying focused.  

## Key Features
- **Distraction Blocking**: Automatically blocks social media sites and messaging apps during focus sessions.  
- **Pomodoro Timer**: Built-in timer to structure your work into productive intervals.  
- **Progress Tracking**: Visualize how many focused hours you’ve accumulated this week.  
- **Cross-Platform**: Available on Windows, macOS, and mobile (iOS/Android).  

## Why Use FocusFlow?
Many productivity apps are either too rigid or too complicated. FocusFlow is simple, intuitive, and directly targets the biggest issue: digital distraction.  

## Setup
1. Download from [focusflow.app](https://focusflow.app).  
2. Install on your device.  
3. Create an account to sync across devices.  

Stay distraction-free and get more done with FocusFlow!"""
agent = AnalyzerAgent()
analyzer_json = agent.run(ANALYZER_PROMPT, doc)
print(analyzer_json)

creators = {
    "EXTRACTOR_PROMPT": ExtractorPromptCreator().build(
        EXTRACTOR_TEMPLATE_EXAMPLE, analyzer_json
    ),
    "INITIAL_SUMMARIZER_PROMPT": SummarizerPromptCreator().build(
        SUMMARIZER_TEMPLATE_EXAMPLE, analyzer_json
    ),
    "TEACHER_PROMPT": TeacherPromptCreator().build(
        TEACHER_TEMPLATE_EXAMPLE, analyzer_json
    ),
    "COMBINE_PROMPT": CombinerPromptCreator().build(
        COMBINER_TEMPLATE_EXAMPLE, analyzer_json
    ),
}
print(ExtractorPromptCreator().build(EXTRACTOR_TEMPLATE_EXAMPLE, analyzer_json))

# Lưu thành file .py
output_file = Path("generated_prompts.py")
with output_file.open("w", encoding="utf-8") as f:
    f.write("# Auto-generated prompts\n\n")
    for name, value in creators.items():
        f.write(f'{name} = """{value}"""\n\n')

print(f"Prompts saved to {output_file}")
