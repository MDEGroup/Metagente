import json
import time
from typing import Any, Dict, List, Optional

from agent.analyzer import AnalyzerAgent
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

#     EXTRACTOR_PROMPT,
#     INITIAL_SUMMARIZER_PROMPT,
#     TEACHER_PROMPT,
# )
from utils.others import get_current_time


def _estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Estimate tokens using the official tokenizer.
    Strict mode for research: if tokenizer not available, raise an error
    instead of falling back to approximation.
    """
    if not isinstance(text, str) or text == "":
        return 0

    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception as e:
        raise RuntimeError(
            f"Token counting failed for model '{model}'. "
            "Please install tiktoken or use the correct tokenizer."
        ) from e


def _count_inputs_token(inputs: Dict[str, Any], model: str = "gpt-4o") -> int:
    total = 0
    for v in inputs.values():
        if isinstance(v, str):
            total += _estimate_tokens(v, model=model)
        elif isinstance(v, list):
            total += sum(
                _estimate_tokens(x, model=model) for x in v if isinstance(x, str)
            )
        elif isinstance(v, dict):
            total += _count_inputs_token(v, model=model)
    return total


class ParallelOptimizer:
    def __init__(self, threshold: float = 0.7, token_model_name: str = "gpt-4o"):
        self.extractor_prompt = EXTRACTOR_PROMPT
        self.summarizer_prompt = INITIAL_SUMMARIZER_PROMPT
        self.teacher_prompt = TEACHER_PROMPT
        self.combiner_prompt = COMBINE_PROMPT
        self.extractor_agent = ExtractorAgent()
        self.teacher_agent = TeacherAgent(TEACHER_PROMPT)
        self.summarizer_agent = SummarizerAgent()
        self.prompt_combine = PromptCombineAgent()
        self.debug_result: Dict[str, Any] = {}
        self.threshold = threshold
        self._token_model_name = token_model_name
        self.analyzer = AnalyzerAgent()

    def _run_step(
        self,
        step_name: str,
        fn,
        *,
        inputs: Dict[str, Any],
        # if_analyzer: bool = False,
        **kwargs,
    ):
        step_start = time.perf_counter()
        ts_iso = get_current_time()

        input_tokens = _count_inputs_token(inputs, model=self._token_model_name)
        output = fn(**kwargs)
        output_str = str(output) if not isinstance(output, str) else output
        output_tokens = _estimate_tokens(output_str, model=self._token_model_name)

        duration = time.perf_counter() - step_start
        metrics = {
            "step": step_name,
            "timestamp": ts_iso,
            "duration_sec": duration,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        # if not if_analyzer:
        #     return output_str, metrics
        # else:
        return output_str, metrics

    def run(
        self,
        max_iterations: int,
        train_data: List[dict],
        log_to_json_path: Optional[str] = None,
    ):
        run_start_ts = get_current_time()
        run_start_perf = time.perf_counter()

        self.debug_result = {
            "timestamp": run_start_ts,
            "Teacher Prompt": self.teacher_prompt,
            "Combine Prompt": self.combiner_prompt,
            "Initial Extractor Prompt": self.extractor_prompt,
            "Initial Summarizer Prompt": self.summarizer_prompt,
        }

        iteration_debug = []
        data_prompt = []
        metrics_all_samples: List[Dict[str, Any]] = []

        for i, data in enumerate(train_data):
            # self.combiner_prompt = CombinerPromptCreator().build(
            #     COMBINER_TEMPLATE_EXAMPLE, analyzer_json
            # )
            # analyzer_json, _ = self._run_step(
            #     "AnalyzerAgent.run",
            #     self.analyzer.run,
            #     inputs={"document": data["description_html_clean"]},
            #     document=data["description_html_clean"],
            #     if_analyzer=True,
            # )

            # slots = prepare_slots(analyzer_json)
            # self.teacher_prompt = fill_prompt(TEACHER_PROMPT, slots)
            # self.extractor_prompt = fill_prompt(EXTRACTOR_PROMPT, slots)
            # self.summarizer_prompt = fill_prompt(INITIAL_SUMMARIZER_PROMPT, slots)
            self.summarizer_prompt = INITIAL_SUMMARIZER_PROMPT
            sample_start_perf = time.perf_counter()
            sample_metrics_steps: List[Dict[str, Any]] = []
            data_debug = []

            description = data["description_short"]
            html = data["description_html_clean"]
            print(f"Data #{i}:\n- Description: {description}")
            print(f" - Extractor Prompt: {self.extractor_prompt}")
            print(f" - Summarizer Prompt: {self.summarizer_prompt}")
            print(f" - Teacher Prompt: {self.teacher_prompt}")

            first_step_inputs = {"prompt": self.extractor_prompt, "readme_text": html}
            first_input_tokens = _count_inputs_token(
                first_step_inputs, model=self._token_model_name
            )

            best_score = 0.0
            best_summarizer_prompt = self.summarizer_prompt

            # Extractor
            extracted_text, step_metrics = self._run_step(
                "ExtractorAgent.run",
                self.extractor_agent.run,
                inputs={"prompt": self.extractor_prompt, "readme_text": html},
                prompt=self.extractor_prompt,
                readme_text=html,
            )
            print(f" Extracted text (first 200 chars): {extracted_text}...")
            sample_metrics_steps.append(step_metrics)
            # extracted_text = data["description_html_clean"]
            max_total_iter = max_iterations
            N, K = 3, 2
            no_improve_count = decrease_count = increase_streak = 0
            max_rougeL_so_far = 0.0
            iter_idx = 0
            last_about_output_tokens = 0

            while iter_idx < max_total_iter:
                about, step_metrics = self._run_step(
                    "SummarizerAgent.run",
                    self.summarizer_agent.run,
                    inputs={
                        "prompt": self.summarizer_prompt,
                        "extracted_text": extracted_text,
                    },
                    prompt=self.summarizer_prompt,
                    extracted_text=extracted_text,
                )
                print(f" Iter {iter_idx} - Generated About: {about}")
                sample_metrics_steps.append(step_metrics)
                last_about_output_tokens = step_metrics["output_tokens"]

                rougeL_score = ROUGE().get_RougeL(string_1=about, string_2=description)
                rouge1_score = ROUGE().get_Rouge1(string_1=about, string_2=description)
                rouge2_score = ROUGE().get_Rouge2(string_1=about, string_2=description)
                print(f" Iter {iter_idx} - ROUGE-L: {rougeL_score:.4f}")

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

                if rougeL_score > max_rougeL_so_far:
                    max_rougeL_so_far = rougeL_score
                    no_improve_count = decrease_count = 0
                    increase_streak += 1
                elif rougeL_score < max_rougeL_so_far:
                    decrease_count += 1
                    no_improve_count += 1
                    increase_streak = 0
                else:
                    no_improve_count += 1
                    increase_streak = 0

                if no_improve_count >= N or decrease_count >= K:
                    if max_rougeL_so_far >= self.threshold:
                        data_prompt.append(best_summarizer_prompt)
                    break

                if rougeL_score < self.threshold:
                    new_prompt, step_metrics = self._run_step(
                        "TeacherAgent.run",
                        self.teacher_agent.run,
                        inputs={
                            "extracted_text": extracted_text,
                            "description": description,
                            "generated_about": about,
                            "rouge_score": str(rougeL_score),
                            "summarizer_prompt": self.summarizer_prompt,
                        },
                        extracted_text=extracted_text,
                        description=description,
                        generated_about=about,
                        rouge_score=rougeL_score,
                        summarizer_prompt=self.summarizer_prompt,
                        prompt=self.teacher_prompt,
                    )
                    sample_metrics_steps.append(step_metrics)
                    self.summarizer_prompt = new_prompt

                else:
                    data_prompt.append(best_summarizer_prompt)
                    break
                iter_idx += 1

            iteration_debug.append(
                {
                    "description_html_clean": html,
                    "description_short": description,
                    "iteration_debug": data_debug,
                    "best_ROUGE-L": best_score,
                    "best_summarizer_prompt": best_summarizer_prompt,
                }
            )

            sample_total_duration = time.perf_counter() - sample_start_perf
            metrics_all_samples.append(
                {
                    "sample_index": i,
                    "first_input_tokens_flow": first_input_tokens,
                    "last_output_tokens_flow": last_about_output_tokens,
                    "total_duration_sec_flow": sample_total_duration,
                    "steps": sample_metrics_steps,
                }
            )

        self.summarizer_prompt, combine_metrics = self._run_step(
            "PromptCombineAgent.run",
            self.prompt_combine.run,
            inputs={"prompt_list": data_prompt},
            prompt_list=data_prompt,
        )
        # self.summarizer_prompt = self.prompt_combine.run(prompt_list=data_prompt)
        metrics_all_samples.append({"combine_summary": combine_metrics})

        run_total_duration = time.perf_counter() - run_start_perf

        self.debug_result["Train Debug"] = iteration_debug
        self.debug_result["Final Extractor Prompt"] = self.extractor_prompt
        self.debug_result["Final Summarizer Prompt"] = self.summarizer_prompt
        self.debug_result["Metrics"] = {
            "run_started_at": run_start_ts,
            "run_total_duration_sec": run_total_duration,
            "samples": metrics_all_samples,
            "aggregate": {
                "num_samples": len(train_data),
                "num_qualified_samples": len(data_prompt),
                "num_disqualified_samples": len(train_data) - len(data_prompt),
            },
        }

        if log_to_json_path:
            with open(log_to_json_path, "w", encoding="utf-8") as f:
                json.dump(self.debug_result, f, ensure_ascii=False, indent=2)
