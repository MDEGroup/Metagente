import argparse
import os
import time
from typing import Any, Dict, List

import pandas as pd
from agent.extractor import ExtractorAgent
from agent.prompt_combine import PromptCombineAgent
from agent.summarizer import SummarizerAgent
from dotenv import load_dotenv
from metric.rouge import ROUGE
from prompt.prompt import EXTRACTOR_PROMPT, OPTIMIZED_SUMMARIZER_PROMPT
from utils.endpoint import MONGODB_COLLECTION, MONGODB_DATABASE, MONGODB_HOST
from utils.mongodb import MongoDB
from utils.others import get_current_time


def _estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Đếm số tokens bằng tiktoken"""
    if not isinstance(text, str) or text == "":
        return 0
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception as e:
        raise RuntimeError(
            f"Token counting failed for model '{model}'. Please install tiktoken."
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


def config_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_file", type=str, required=True)
    parser.add_argument("--test_result_dir", type=str, required=True)
    parser.add_argument("--use_mongodb", action="store_false", required=False)
    args = parser.parse_args()
    return args


class Evaluation:
    def __init__(self, token_model_name: str = "gpt-4o"):
        self.extractor_prompt = EXTRACTOR_PROMPT
        self.summarizer_prompt = OPTIMIZED_SUMMARIZER_PROMPT
        self.extractor_agent = ExtractorAgent()
        self.summarizer_agent = SummarizerAgent()
        self.prompt_combine = PromptCombineAgent()
        self._token_model_name = token_model_name
        self.debug_result = {}

    def _run_step(
        self,
        step_name: str,
        fn,
        *,
        inputs: Dict[str, Any],
        **kwargs,
    ):
        step_start = time.perf_counter()
        ts_iso = get_current_time()

        input_tokens = _count_inputs_token(inputs, model=self._token_model_name)
        output = fn(**kwargs)
        output_str = output if isinstance(output, str) else str(output)
        output_tokens = _estimate_tokens(output_str, model=self._token_model_name)

        duration = time.perf_counter() - step_start
        metrics = {
            "step": step_name,
            "timestamp": ts_iso,
            "duration_sec": duration,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        return output_str, metrics

    def run(self, dataset: list[dict]):
        self.debug_result = {
            "timestamp": get_current_time(),
            "Extractor Prompt": self.extractor_prompt,
            "Summarizer Prompt": self.summarizer_prompt,
            "metrics_all_samples": [],
        }

        data_debug = []
        total_rougeL = total_rouge1 = total_rouge2 = 0
        num_data = 0

        for i, data in enumerate(dataset):
            try:
                sample_metrics_steps: List[Dict[str, Any]] = []
                sample_start_perf = time.perf_counter()

                description = data["description_short"]
                html = data["description_html_clean"]
                print(f"Data #{i}:\n- Description: {description}")

                # --- Extractor step ---
                extracted_text, step_metrics = self._run_step(
                    "ExtractorAgent.run",
                    self.extractor_agent.run,
                    inputs={"prompt": self.extractor_prompt, "readme_text": html},
                    prompt=self.extractor_prompt,
                    readme_text=html,
                )
                sample_metrics_steps.append(step_metrics)

                # --- Summarizer step ---
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
                sample_metrics_steps.append(step_metrics)

                print(f"Generated About: {about}")

                # ROUGE scores
                rougeL_score = ROUGE().get_RougeL(string_1=about, string_2=description)
                rouge1_score = ROUGE().get_Rouge1(string_1=about, string_2=description)
                rouge2_score = ROUGE().get_Rouge2(string_1=about, string_2=description)

                total_rougeL += rougeL_score
                total_rouge1 += rouge1_score
                total_rouge2 += rouge2_score
                num_data += 1

                print(
                    f"Rouge1: {rouge1_score}, Rouge2: {rouge2_score}, RougeL: {rougeL_score}"
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
                    }
                )

                # sample summary metrics
                sample_total_duration = time.perf_counter() - sample_start_perf
                self.debug_result["metrics_all_samples"].append(
                    {
                        "sample_index": i,
                        "steps": sample_metrics_steps,
                        "total_duration_sec": sample_total_duration,
                    }
                )

            except Exception as e:
                print(f"Error while running data {i}: {e}")
                continue

        # averages
        avg_rougeL_score = total_rougeL / num_data if num_data else 0
        avg_rouge1_score = total_rouge1 / num_data if num_data else 0
        avg_rouge2_score = total_rouge2 / num_data if num_data else 0

        print(f"Avg RougeL: {avg_rougeL_score}")
        print(f"Avg Rouge1: {avg_rouge1_score}")
        print(f"Avg Rouge2: {avg_rouge2_score}")

        self.debug_result.update(
            {
                "data_length": num_data,
                "data_debug": data_debug,
                "total_rouge1_score": total_rouge1,
                "avg_rouge1_score": avg_rouge1_score,
                "total_rouge2_score": total_rouge2,
                "avg_rouge2_score": avg_rouge2_score,
                "total_rougeL_score": total_rougeL,
                "avg_rougeL_score": avg_rougeL_score,
            }
        )


class Main:
    def __init__(self):
        self.args = config_parser()
        if self.args.use_mongodb:
            self.debug_db = MongoDB(
                host=MONGODB_HOST,
                database=MONGODB_DATABASE,
                collection=MONGODB_COLLECTION,
            )

    def run(self):
        test_data = pd.read_csv(self.args.test_data_file).to_dict(orient="records")

        # chạy flow 3 lần
        for run_id in range(1, 3):  # chạy đủ 3 lần
            print("\n============================")
            print(f"▶️ Flow run {run_id}")
            print("============================")

            evaluation = Evaluation()
            evaluation.run(test_data)

            if self.args.use_mongodb:
                result_with_id = dict(evaluation.debug_result)
                result_with_id["run_id"] = run_id
                self.debug_db.add_data(result_with_id)

            # ---- Lưu CSV ----
            data_debug = evaluation.debug_result.get("data_debug", [])
            if data_debug:
                df = pd.DataFrame(data_debug)
                out_csv = os.path.join(self.args.test_result_dir, f"run_{run_id}.csv")
                df.to_csv(out_csv, index=False, encoding="utf-8")
                print(f"💾 Saved CSV results to {out_csv}")
            else:
                print(f"⚠️ Không có data_debug để lưu CSV cho run {run_id}")


if __name__ == "__main__":
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
    main_flow = Main()
    main_flow.run()
