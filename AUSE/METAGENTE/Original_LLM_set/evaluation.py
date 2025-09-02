import argparse
import os

import pandas as pd
from agent.extractor import ExtractorAgent
from agent.summarizer import SummarizerAgent
from dotenv import load_dotenv
from metric.rouge import ROUGE
from prompt.prompt import EXTRACTOR_PROMPT, OPTIMIZED_SUMMARIZER_PROMPT
from utils.endpoint import MONGODB_COLLECTION, MONGODB_DATABASE, MONGODB_HOST
from utils.mongodb import MongoDB
from utils.others import get_current_time, save_evaluation_result


def config_parser() -> argparse.Namespace:
    """
    Add configuration arguments passed in the command line, including:
        - test_data_file: The path to the test data file
        - test_result_dir: The directory to save the test results
        - use_mongodb: Whether to use MongoDB for storing debug data

    Returns:
        argparse.Namespace: The parsed arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_file", type=str, required=True)
    parser.add_argument("--test_result_dir", type=str, required=True)
    parser.add_argument("--use_mongodb", action="store_false", required=False)
    args = parser.parse_args()
    return args


class Evaluation:
    def __init__(self):
        self.extractor_prompt = EXTRACTOR_PROMPT
        self.summarizer_prompt = OPTIMIZED_SUMMARIZER_PROMPT
        self.extractor_agent = ExtractorAgent()
        self.summarizer_agent = SummarizerAgent()
        self.debug_result = {}

    def run(self, dataset: list[dict]):
        self.debug_result = {
            "timestamp": get_current_time(),
            "Extractor Prompt": self.extractor_prompt,
            "Summarizer Prompt": self.summarizer_prompt,
        }

        data_debug = []
        total_rougeL = 0
        total_rouge1 = 0
        total_rouge2 = 0
        num_data = 0
        for i, data in enumerate(dataset):
            try:
                description = data["description_short"]
                html = data["description_html_clean"]
                print(f"Data #{i}:\n- Description: {description}")

                extracted_text = self.extractor_agent.run(
                    prompt=self.extractor_prompt, readme_text=html
                )

                about = self.summarizer_agent.run(
                    prompt=self.summarizer_prompt, extracted_text=extracted_text
                )

                print(f"Generated About: {about}")

                rougeL_score = ROUGE().get_RougeL(string_1=about, string_2=description)
                total_rougeL += rougeL_score
                rouge1_score = ROUGE().get_Rouge1(string_1=about, string_2=description)
                total_rouge1 += rouge1_score
                rouge2_score = ROUGE().get_Rouge2(string_1=about, string_2=description)
                total_rouge2 += rouge2_score

                print(f"Rouge1 Score: {rouge1_score}")
                print(f"Rouge2 Score: {rouge2_score}")
                print(f"RougeL Score: {rougeL_score}")

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
                num_data += 1
            except Exception as e:
                print(f"Error while running data {i}: {e}")
                continue

        avg_rougeL_score = total_rougeL / num_data
        avg_rouge1_score = total_rouge1 / num_data
        avg_rouge2_score = total_rouge2 / num_data

        print(f"Avg RougeL Score: {avg_rougeL_score}")
        print(f"Avg Rouge1 Score: {avg_rouge1_score}")
        print(f"Avg Rouge2 Score: {avg_rouge2_score}")

        self.debug_result["data_length"] = num_data
        self.debug_result["data_debug"] = data_debug
        self.debug_result["total_rouge1_score"] = total_rouge1
        self.debug_result["avg_rouge1_score"] = avg_rouge1_score
        self.debug_result["total_rouge2_score"] = total_rouge2
        self.debug_result["avg_rouge2_score"] = avg_rouge2_score
        self.debug_result["total_rougeL_score"] = total_rougeL
        self.debug_result["avg_rougeL_score"] = avg_rougeL_score


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
        evaluation = Evaluation()
        evaluation.run(test_data)

        if self.args.use_mongodb:
            self.debug_db.add_data(evaluation.debug_result)
        save_evaluation_result(self.args.test_result_dir, evaluation.debug_result)


if __name__ == "__main__":
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
    main_flow = Main()
    main_flow.run()
