import csv
import os
from datetime import datetime as dt


def get_current_time() -> float:
    """Get current time in UNIX timestamp"""
    return dt.now().timestamp()


def _extract_good_data(debug_result: dict) -> dict:
    good_data = []

    for data in debug_result["Train Debug"]:
        if data["best_ROUGE-L"] >= 0.7:
            good_data.append(data)

    return good_data


def save_parallel_train_result(train_result_dir: str, debug_result: dict) -> None:
    optimized_prompt = '"""' + debug_result["Final Summarizer Prompt"] + '"""'

    with open("prompt/prompt.py", "r") as file:
        lines = file.readlines()

    found_optimized_prompt = False
    for i, line in enumerate(lines):
        if line.strip().startswith("OPTIMIZED_SUMMARIZER_PROMPT ="):
            lines = lines[: i + 1]
            lines[i] = f"""OPTIMIZED_SUMMARIZER_PROMPT = "{optimized_prompt}"\n"""
            found_optimized_prompt = True
            break

    if not found_optimized_prompt:
        lines.append(f"""\nOPTIMIZED_SUMMARIZER_PROMPT = {optimized_prompt}\n""")

    with open("prompt/prompt.py", "w") as file:
        file.writelines(lines)

    good_data_result = _extract_good_data(debug_result)

    train_result = []

    for data_id, data in enumerate(good_data_result):
        single_data = []
        for i, iter_data in enumerate(data["iteration_debug"]):
            single_data.append(
                [
                    data_id,
                    i,
                    iter_data["extracted_text"],
                    iter_data["summarizer_prompt"],
                    iter_data["generated_about"],
                    iter_data["rouge1_score"],
                    iter_data["rouge2_score"],
                    iter_data["rougeL_score"],
                ]
            )

        single_data[0].extend(
            [data["description_html_clean"], data["description_short"]]
        )

        train_result.extend(single_data)

    train_result.append([None] * 10 + [debug_result["Final Summarizer Prompt"]])

    header = [
        "Data ID",
        "Iteration",
        "Extracted text from Extractor Agent",
        "Prompt used for Summarizer Agent",
        "Generated About",
        "ROUGE-1 score",
        "ROUGE-2 score",
        "ROUGE-L score",
        "HTML",
        "Ground truth description",
        "Final Summarizer Prompt",
    ]

    with open(
        os.path.join(train_result_dir, "train_result.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row if provided
        if header:
            writer.writerow(header)

        # Write the data rows
        writer.writerows(train_result)


def save_sequential_train_result(train_result_dir: str, debug_result: dict) -> None:
    optimized_prompt = '"""' + debug_result["Best Summarizer Prompt"] + '"""'

    with open("prompt/prompt.py", "r") as file:
        lines = file.readlines()

    found_optimized_prompt = False
    for i, line in enumerate(lines):
        if line.strip().startswith("OPTIMIZED_SUMMARIZER_PROMPT ="):
            lines = lines[: i + 1]
            lines[i] = f"""OPTIMIZED_SUMMARIZER_PROMPT = "{optimized_prompt}"\n"""
            found_optimized_prompt = True
            break

    if not found_optimized_prompt:
        lines.append(f"""\nOPTIMIZED_SUMMARIZER_PROMPT = {optimized_prompt}\n""")

    with open("prompt/prompt.py", "w") as file:
        file.writelines(lines)

    train_result = []

    prompt = debug_result["Initial Summarizer Prompt"]

    for iter_id, iter_value in enumerate(debug_result["Iteration Debug"]):
        single_iteration = []
        for data_id, data_value in enumerate(iter_value["data_debug"]):
            if iter_id == 0:
                html = data_value["description_html_clean"]
                description = data_value["description_short"]
            else:
                html = None
                description = None

            single_iteration.append(
                [
                    iter_id,
                    data_id,
                    None,
                    html,
                    description,
                    data_value["extracted_text"],
                    data_value["generated_about"],
                    data_value["rouge1_score"],
                    data_value["rouge2_score"],
                    data_value["rougeL_score"],
                    data_value["analysis_reasoning"],
                ]
            )

        single_iteration[0][2] = prompt
        single_iteration[0].append(iter_value["analysis_summary"])

        train_result.extend(single_iteration)

        prompt = iter_value["new_summarizer_prompt"]

    train_result.append([None] * 12 + [debug_result["Best Summarizer Prompt"]])

    header = [
        "Iteration",
        "Data ID",
        "Prompt used for Summarizer Agent",
        "HTML",
        "Ground truth description",
        "Extracted text from Extractor Agent",
        "Generated About",
        "ROUGE-1 score",
        "ROUGE-2 score",
        "ROUGE-L score",
        "Analysis Reasoning",
        "Analysis Summary",
        "Final Summarizer Prompt",
    ]

    with open(
        os.path.join(train_result_dir, "train_result.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row if provided
        if header:
            writer.writerow(header)

        # Write the data rows
        writer.writerows(train_result)


def save_evaluation_result(test_result_dir: str, debug_result: dict) -> None:
    csv_data = []

    for data in debug_result["data_debug"]:
        csv_data.append(
            [
                data["description_short"],
                data["generated_about"],
                data["rouge1_score"],
                data["rouge2_score"],
                data["rougeL_score"],
            ]
        )

    csv_data[0].extend(
        [
            debug_result["avg_rouge1_score"],
            debug_result["avg_rouge2_score"],
            debug_result["avg_rougeL_score"],
        ]
    )

    header = [
        "Description",
        "Generated About",
        "ROUGE-1",
        "ROUGE-2",
        "ROUGE-L",
        "Average ROUGE-1",
        "Average ROUGE-2",
        "Average ROUGE-L",
    ]

    with open(
        os.path.join(test_result_dir, "test_result.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row if provided
        if header:
            writer.writerow(header)

        # Write the data rows
        writer.writerows(csv_data)
