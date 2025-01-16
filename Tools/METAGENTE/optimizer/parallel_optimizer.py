from agent.extractor import ExtractorAgent
from agent.prompt_combine import PromptCombineAgent
from agent.summarizer import SummarizerAgent
from agent.teacher import TeacherAgent
from metric.rouge import ROUGE
from prompt.prompt import (
    COMBINE_PROMPT,
    EXTRACTOR_PROMPT,
    INITIAL_SUMMARIZER_PROMPT,
    TEACHER_PROMPT,
)
from utils.others import get_current_time


class ParallelOptimizer:
    def __init__(self, threshold: float = 0.7):
        self.extractor_prompt = EXTRACTOR_PROMPT
        self.summarizer_prompt = INITIAL_SUMMARIZER_PROMPT
        self.extractor_agent = ExtractorAgent()
        self.teacher_agent = TeacherAgent()
        self.summarizer_agent = SummarizerAgent()
        self.prompt_combine = PromptCombineAgent()
        self.debug_result = {}
        self.threshold = threshold

    def run(
        self,
        max_iterations: int,
        train_data: list[dict],
    ):
        self.debug_result = {
            "timestamp": get_current_time(),
            "Teacher Prompt": TEACHER_PROMPT,
            "Combine Prompt": COMBINE_PROMPT,
            "Initial Extractor Prompt": self.extractor_prompt,
            "Initial Summarizer Prompt": self.summarizer_prompt,
        }

        iteration_debug = []
        data_prompt = []

        for i, data in enumerate(train_data):
            data_debug = []
            self.summarizer_prompt = INITIAL_SUMMARIZER_PROMPT

            description = data["description"]
            readme = data["readme"]
            print(f"Data #{i}:\n- Description: {description}")

            best_score = 0
            best_summarizer_prompt = self.summarizer_prompt

            extracted_text = self.extractor_agent.run(
                prompt=self.extractor_prompt, readme_text=readme
            )

            for iter in range(max_iterations):
                print(f"Iteration #{iter}:")

                print(f"Extracted Text: {extracted_text}")

                about = self.summarizer_agent.run(
                    prompt=self.summarizer_prompt, extracted_text=extracted_text
                )

                print(f"Generated About: {about}")

                rougeL_score = ROUGE().get_RougeL(string_1=about, string_2=description)
                rouge1_score = ROUGE().get_Rouge1(string_1=about, string_2=description)
                rouge2_score = ROUGE().get_Rouge2(string_1=about, string_2=description)

                print(f"Rouge1 Score: {rouge1_score}")
                print(f"Rouge2 Score: {rouge2_score}")
                print(f"RougeL Score: {rougeL_score}")

                turn_debug = {
                    "summarizer_prompt": self.summarizer_prompt,
                    "readme": readme,
                    "extracted_text": extracted_text,
                    "description": description,
                    "generated_about": about,
                    "rouge1_score": rouge1_score,
                    "rouge2_score": rouge2_score,
                    "rougeL_score": rougeL_score,
                }

                data_debug.append(turn_debug)

                if rougeL_score > best_score:
                    best_score = rougeL_score
                    best_summarizer_prompt = self.summarizer_prompt

                if rougeL_score < self.threshold:
                    self.summarizer_prompt = self.teacher_agent.run(
                        extracted_text=extracted_text,
                        description=description,
                        generated_about=about,
                        rouge_score=rougeL_score,
                        summarizer_prompt=self.summarizer_prompt,
                    )

                    print(f"New Summarizer Prompt: {self.summarizer_prompt}")
                else:
                    data_prompt.append(best_summarizer_prompt)
                    break

            print(f"Best RougeL Score for Data #{i}: {best_score}")

            iteration_debug.append(
                {
                    "readme": readme,
                    "description": description,
                    "iteration_debug": data_debug,
                    "best_ROUGE-L": best_score,
                    "best_summarizer_prompt": best_summarizer_prompt,
                }
            )

        self.summarizer_prompt = self.prompt_combine.run(prompt_list=data_prompt)

        print(f"Final Result:\nSummarizer Prompt: {self.summarizer_prompt}")

        self.debug_result["Train Debug"] = iteration_debug
        self.debug_result["Final Extractor Prompt"] = self.extractor_prompt
        self.debug_result["Final Summarizer Prompt"] = self.summarizer_prompt
