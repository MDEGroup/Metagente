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

    def run(self, max_iterations: int, train_data: list[dict]):
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

            description = data["description_short"]
            html = data["description_html_clean"]
            print(f"Data #{i}:\n- Description: {description}")

            best_score = 0
            best_summarizer_prompt = self.summarizer_prompt

            extracted_text = self.extractor_agent.run(
                prompt=self.extractor_prompt, readme_text=html
            )

            max_total_iter = max_iterations
            # N = 3
            # K = 2
            # X = 2
            # extend_step = 3

            # no_improve_count = 0
            # decrease_count = 0
            # increase_streak = 0
            # max_rougeL_so_far = 0
            iter = 0

            while iter < max_total_iter:
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
                    "description_html_clean": html,
                    "extracted_text": extracted_text,
                    "description_short": description,
                    "generated_about": about,
                    "rouge1_score": rouge1_score,
                    "rouge2_score": rouge2_score,
                    "rougeL_score": rougeL_score,
                }

                data_debug.append(turn_debug)

                if rougeL_score > best_score:
                    best_score = rougeL_score
                    best_summarizer_prompt = self.summarizer_prompt

                # if rougeL_score > max_rougeL_so_far:
                #     delta = rougeL_score - max_rougeL_so_far
                #     max_rougeL_so_far = rougeL_score
                #     no_improve_count = 0
                #     decrease_count = 0
                #     increase_streak += 1

                # elif rougeL_score < max_rougeL_so_far:
                #     decrease_count += 1
                #     no_improve_count += 1
                #     increase_streak = 0

                # else:
                #     no_improve_count += 1
                #     increase_streak = 0

                # if no_improve_count >= N or decrease_count >= K:
                #     if max_rougeL_so_far >= self.threshold:
                #         data_prompt.append(best_summarizer_prompt)
                #     print("Stopping early due to no improvement or consistent drop.")
                #     break
                if rougeL_score < self.threshold:
                    self.summarizer_prompt = self.teacher_agent.run(
                        extracted_text=extracted_text,
                        description=description,
                        generated_about=about,
                        rouge_score=rougeL_score,
                        summarizer_prompt=self.summarizer_prompt,
                    )
                else:
                    data_prompt.append(best_summarizer_prompt)
                    break
                print(f"New Summarizer Prompt: {self.summarizer_prompt}")

                iter += 1

            print(f"Best RougeL Score for Data #{i}: {best_score}")

            iteration_debug.append(
                {
                    "description_html_clean": html,
                    "description_short": description,
                    "iteration_debug": data_debug,
                    "best_ROUGE-L": best_score,
                    "best_summarizer_prompt": best_summarizer_prompt,
                }
            )

        self.summarizer_prompt = self.prompt_combine.run(prompt_list=data_prompt)

        print(f"Final Summarizer Prompt: {self.summarizer_prompt}")
        print(f"Number of disqualified samples: {len(train_data) - len(data_prompt)}")

        self.debug_result["Train Debug"] = iteration_debug
        self.debug_result["Final Extractor Prompt"] = self.extractor_prompt
        self.debug_result["Final Summarizer Prompt"] = self.summarizer_prompt


# TO USE DYNAMIC FLOW: UNCOMMENT ALL THE CODES
