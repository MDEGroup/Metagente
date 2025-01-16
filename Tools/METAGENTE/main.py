import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from optimizer.parallel_optimizer import ParallelOptimizer
from optimizer.sequential_optimizer import SequentialOptimizer
from utils.endpoint import MONGODB_COLLECTION, MONGODB_DATABASE, MONGODB_HOST
from utils.mongodb import MongoDB
from utils.others import save_parallel_train_result, save_sequential_train_result


def config_parser() -> argparse.Namespace:
    """
    Add configuration arguments passed in the command line, including:
        - train_data_file: The path to the training data file
        - train_result_dir: The directory to save the training results
        - optimizer: The optimizer to use between sequential and data parallel optimizer
        - parallel_threshold: The optimized prompt's result threshold for data acceptance when using data parallel optimizer
        - use_mongodb: Whether to use MongoDB for storing debug data

    Returns:
        argparse.Namespace: The parsed arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file", type=str, required=True)
    parser.add_argument("--train_result_dir", type=str, required=True)
    parser.add_argument(
        "--optimizer",
        default="data_parallel",
        type=str,
        choices=["sequential", "data_parallel"],
        required=False,
    )
    parser.add_argument("--num_iterations", default=15, type=int, required=False)
    parser.add_argument("--parallel_threshold", default=0.7, type=float, required=False)
    parser.add_argument("--use_mongodb", action="store_false", required=False)
    args = parser.parse_args()
    return args


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
        train_data = pd.read_csv(self.args.train_data_file).to_dict(orient="records")

        optimizer = ParallelOptimizer()
        if self.args.optimizer == "sequential":
            optimizer = SequentialOptimizer()

        optimizer.run(self.args.num_iterations, train_data)

        if self.args.use_mongodb:
            self.debug_db.add_data(optimizer.debug_result)
        if self.args.optimizer == "data_parallel":
            save_parallel_train_result(
                self.args.train_result_dir, optimizer.debug_result
            )
        elif self.args.optimizer == "sequential":
            save_sequential_train_result(
                self.args.train_result_dir, optimizer.debug_result
            )


if __name__ == "__main__":
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
    main_flow = Main()
    main_flow.run()
