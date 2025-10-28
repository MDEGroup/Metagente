# metagente

This repository contains the source code implementation of metagente and the datasets used to replicate the experimental results

## Getting Started

### Dependencies

* Python version 3.10.12
* Python packages are listed in `requirements.txt`
* MongoDB

### Running the application

Installing necessary packages:
```Python
pip install -r requirements.txt
```

Running the optimization code:
```Python
python main.py \
--train_data_file data/train_data.csv \
--train_result_dir result/parallel \
```

Running the evaluation code:
```Python
python evaluation.py \
--test_data_file data/test_data.csv \
--test_result_dir result/parallel 
```
(test set 400/600/top_200 Rouge/Cosine can be tested by replacing test_data.csv with respective files)