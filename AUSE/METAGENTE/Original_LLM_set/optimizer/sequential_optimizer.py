from agent.analysis import AnalysisAgent
from agent.analysis_summarizer import AnalysisSummarizerAgent
from agent.extractor import ExtractorAgent
from agent.sequential_teacher import SequentialTeacherAgent
from agent.summarizer import SummarizerAgent
from metric.rouge import ROUGE
from prompt.prompt import (
    ANALYSIS_PROMPT,
    ANALYSIS_SUMMARIZER_PROMPT,
    EXTRACTOR_PROMPT,
    INITIAL_SUMMARIZER_PROMPT,
    SEQUENTIAL_TEACHER_PROMPT,
)
from utils.others import get_current_time


class SequentialOptimizer:
    def __init__(self):
        self.extractor_prompt = EXTRACTOR_PROMPT
        self.summarizer_prompt = INITIAL_SUMMARIZER_PROMPT
        self.extractor_agent = ExtractorAgent()
        self.teacher_agent = SequentialTeacherAgent()
        self.summarizer_agent = SummarizerAgent()
        self.analysis_agent = AnalysisAgent()
        self.analysis_summarizer = AnalysisSummarizerAgent()
        self.debug_result = {}

    def run(
        self,
        max_iterations: int,
        train_data: list[dict],
    ):
        self.debug_result = {
            "timestamp": get_current_time(),
            "Analysis Prompt": ANALYSIS_PROMPT,
            "Analysis Summarizer Prompt": ANALYSIS_SUMMARIZER_PROMPT,
            "Teacher Prompt": SEQUENTIAL_TEACHER_PROMPT,
            "Initial Extractor Prompt": self.extractor_prompt,
            "Initial Summarizer Prompt": self.summarizer_prompt,
        }

        iteration_debug = []
        best_score = 0
        best_analysis_summary = ""
        best_extractor_prompt = self.extractor_prompt
        best_summarizer_prompt = self.summarizer_prompt
        worst_score = 1
        worst_analysis_summary = ""
        worst_summarizer_prompt = self.summarizer_prompt
        for iteration in range(max_iterations):
            print(f"Iteration #{iteration}:")
            iteration_analysis = []
            data_debug = []
            total_rouge_score = 0
            total_rouge1 = 0
            total_rouge2 = 0
            for id, data in enumerate(train_data):
                description = data["description_short"]
                html = data["description_html_clean"]
                print(f"Data #{id}:\n- Description: {description}")

                extracted_text = self.extractor_agent.run(
                    prompt=self.extractor_prompt, readme_text=html
                )

                print(f"Extracted Text: {extracted_text}")

                about = self.summarizer_agent.run(
                    prompt=self.summarizer_prompt, extracted_text=extracted_text
                )

                print(f"Generated About: {about}")

                rougeL_score = ROUGE().get_RougeL(string_1=about, string_2=description)
                total_rouge_score += rougeL_score
                rouge1_score = ROUGE().get_Rouge1(string_1=about, string_2=description)
                total_rouge1 += rouge1_score
                rouge2_score = ROUGE().get_Rouge2(string_1=about, string_2=description)
                total_rouge2 += rouge2_score

                print(f"Rouge1 Score: {rouge1_score}")
                print(f"Rouge2 Score: {rouge2_score}")
                print(f"RougeL Score: {rougeL_score}")

                analysis_reasoning = self.analysis_agent.run(
                    generated_about=about, ground_truth=description, score=rougeL_score
                )

                print(f"Analysis Reasoning: {analysis_reasoning}\n")

                iteration_analysis.append(
                    {
                        "description_short": description,
                        "generated_about": about,
                        "analysis_reasoning": analysis_reasoning,
                    }
                )

                data_debug.append(
                    {
                        "description_html_clean": html,
                        "extracted_text": extracted_text,
                        "description_short": description,
                        "generated_about": about,
                        "rouge1_score": rouge1_score,
                        "rouge2_score": rouge2_score,
                        "rougeL_score": rougeL_score,
                        "analysis_reasoning": analysis_reasoning,
                    }
                )

            analysis_summary = self.analysis_summarizer.run(
                analysis_result=iteration_analysis
            )

            print(f"Analysis Summary: {analysis_summary}")

            avg_rouge_score = total_rouge_score / len(train_data)
            avg_rouge1_score = total_rouge1 / len(train_data)
            avg_rouge2_score = total_rouge2 / len(train_data)

            print(f"Avg RougeL Score: {avg_rouge_score}")
            print(f"Avg Rouge1 Score: {avg_rouge1_score}")
            print(f"Avg Rouge2 Score: {avg_rouge2_score}")

            if avg_rouge_score > best_score:
                best_score = avg_rouge_score
                best_analysis_summary = analysis_summary
                best_extractor_prompt = self.extractor_prompt
                best_summarizer_prompt = self.summarizer_prompt
                best_rouge1 = avg_rouge1_score
                best_rouge2 = avg_rouge2_score

            if avg_rouge_score < worst_score:
                worst_score = avg_rouge_score
                worst_analysis_summary = analysis_summary
                worst_summarizer_prompt = self.summarizer_prompt

            self.summarizer_prompt = self.teacher_agent.run(
                best_summarizer_prompt=best_summarizer_prompt,
                best_analysis_summary=best_analysis_summary,
                best_score=best_score,
                worst_summarizer_prompt=worst_summarizer_prompt,
                worst_analysis_summary=worst_analysis_summary,
                worst_score=worst_score,
                current_summarizer_prompt=self.summarizer_prompt,
                current_analysis_summary=analysis_summary,
            )

            print(
                f"After Iteration #{iteration}:\nNew Extractor Prompt: {self.extractor_prompt}\nNew Summarizer Prompt: {self.summarizer_prompt}\n"
            )

            iteration_debug.append(
                {
                    "data_debug": data_debug,
                    "analysis_summary": analysis_summary,
                    "avg_rouge1_score": avg_rouge1_score,
                    "avg_rouge2_score": avg_rouge2_score,
                    "avg_rougeL_score": avg_rouge_score,
                    "new_extractor_prompt": self.extractor_prompt,
                    "new_summarizer_prompt": self.summarizer_prompt,
                }
            )

        print(
            f"Final Result:\nBest Avg Rouge1 Score: {best_rouge1}\nBest Avg Rouge2 Score: {best_rouge2}\nBest Avg RougeL Score: {best_score}\nBest Extractor Prompt: {best_extractor_prompt}\nBest Summarizer Prompt: {best_summarizer_prompt}"
        )

        self.debug_result["Iteration Debug"] = iteration_debug
        self.debug_result["Best Avg Rouge1 Score"] = best_rouge1
        self.debug_result["Best Avg Rouge2 Score"] = best_rouge2
        self.debug_result["Best Avg RougeL Score"] = best_score
        self.debug_result["Best Extractor Prompt"] = best_extractor_prompt
        self.debug_result["Best Summarizer Prompt"] = best_summarizer_prompt
