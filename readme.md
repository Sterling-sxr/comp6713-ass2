# COMP9417 Group Project:Detecting AI-Generated Text using Linguistic Features and Transformer Models

## 1. Project Overview

This project focuses on detecting AI-generated text through multiple modelling approaches. The task is formulated as a binary text classification problem, where the goal is to distinguish between human-written text and AI-generated text.

To study this problem, we implemented and compared five different approaches:

- **Model 01:** Baseline Method
- **Model 02:** Existing AI Detector
- **Model 03:** Existing BERT Detector
- **Model 04:** Fine-tuned Model
- **Model 05:** Extended Method

In addition to model development, the repository also includes data preprocessing, model re-evaluation, and a demo module.

## 2.  Repository Structure

```text
.
├── .git/                           # Git version control files
├── data/                           # Main dataset directory
├── datasets/                       # Additional datasets or processed data
├── demo/                           # Demonstration module
├── model01_baseline_method/        # Baseline model implementation
├── model02_existing_ai_detector/   # Existing public AI-text detector
├── model03_existing_bert_detector/ # Existing BERT-based detector
├── model04_finetuned_model/        # Fine-tuned model
├── model05_extend_a_method/        # Extended / improved method
├── .gitignore                      # Files ignored by Git
├── data_preprocessing              # Data preprocessing file
└── re_evaluate_all_models          # File for re-evaluating all models
```
## 3. Methods
Model 01: Baseline Method

This is the baseline approach used for comparison with more advanced models. It provides a simple and interpretable reference point for the project.

Model 02: Existing AI Detector

This model uses an existing public AI-text detector. It allows us to evaluate how well an off-the-shelf detector performs on our dataset.

Model 03: Existing BERT Detector

This model uses an existing BERT-based text detector. It allows us to evaluate how well a transformer-based detector performs on our dataset without additional fine-tuning.

Model 04: Fine-tuned Model

This approach fine-tunes a transformer model on the project dataset. The goal is to improve performance by adapting the model to the specific classification task.

Model 05: Extended Method

This model extends one of the previous approaches. It is designed to explore whether additional modifications can further improve detection performance.

## 4. Dataset

The dataset for this project is stored in the data/ and/or datasets/ directories.

A typical data split structure is as follows:

data/
└── splits/
    ├── train.parquet
    ├── val.parquet
    └── test.parquet

Please make sure all required dataset files are placed in the correct directories before running the project.

## 5. Environment Requirements

Recommended environment:

Python 3.10 or above
pip or conda
GPU recommended for transformer-based models

Common packages used in this project include:

pandas
numpy
scikit-learn
matplotlib
torch
transformers
pyarrow

Install dependencies with:

pip install -r requirements.txt
## 6. How to Run
Step 1: Data Preprocessing

Run the preprocessing file first to prepare the dataset.

python data_preprocessing

If the preprocessing is provided as a notebook in your local setup, open it in Jupyter Notebook, JupyterLab, or VS Code and run all cells.

Step 2: Run Individual Models

Most model implementations in this repository are provided as Jupyter notebooks. Please open and run the following notebooks:

model01_baseline_method/model01_baseline.ipynb
model02_existing_ai_detector/model02_existing_ai_detector.ipynb
model03_existing_bert_detector/model03_existing_bert_detector.ipynb
model04_finetuned_model/model4.ipynb
model05_extend_a_method/extend_a_method.ipynb

These notebooks can be run in:

Jupyter Notebook
JupyterLab
VS Code
Step 3: Re-evaluate All Models

To compare the final performance of all models, run:

python re_evaluate_all_models

If the re-evaluation is provided as a notebook in your local setup, open it and run all cells accordingly.

## 7. Evaluation Metrics

The models in this project are evaluated using standard classification metrics, including:

Accuracy
Precision
Recall
F1-score

These metrics are used to compare the effectiveness of different approaches on the AI-text detection task.

## 8. Outputs and Results

The outputs of each model are stored in their corresponding folders.

Model 01: Baseline Method

Stored in model01_baseline_method/:

baseline_model.joblib: saved baseline model
model01_baseline.ipynb: notebook for baseline implementation and evaluation
Model 02: Existing AI Detector

Stored in model02_existing_ai_detector/:

model02_existing_ai_detector.ipynb: notebook for running the existing AI detector
existing_model_test_metrics.json: evaluation metrics on the test set
existing_model_test_predictions.csv: full test predictions
existing_model_test_predictions_compact.csv: compact version of test predictions
Model 03: Existing BERT Detector

Stored in model03_existing_bert_detector/:

model03_existing_bert_detector.ipynb: notebook for running the existing BERT-based detector
existing_bert_model_test_metrics.json: evaluation metrics on the test set
existing_bert_model_test_predictions.csv: full test predictions
existing_bert_model_test_predictions_compact.csv: compact version of test predictions
Model 04: Fine-tuned Model

Stored in model04_finetuned_model/:

model4.ipynb: notebook for fine-tuning and evaluation
results/: directory containing outputs and result files
Model 05: Extended Method

Stored in model05_extend_a_method/:

extend_a_method.ipynb: notebook for the extended method

Please refer to each folder for the corresponding notebooks, saved models, predictions, metrics, and result files.

## 9. Reproducibility

To reproduce the experiments in this repository:

Prepare the dataset in the expected folder structure.
Run the data preprocessing file.
Run each model notebook separately.
Re-evaluate all models for final comparison.

Make sure file paths are correct relative to your current working directory.

## 10. Notes
Some large datasets or model files may not be included in the repository due to size limitations.
Relative paths may need adjustment depending on where the files are executed.
GPU is strongly recommended for BERT-based and fine-tuned transformer models.
## 11. Team Contributions

This project was completed as a group project.

Team Members
Runming Zhang — z5546629@ad.unsw.edu.au
Tianhao Hua — z5536343@ad.unsw.edu.au
Bobby Zhang — z5511950@ad.unsw.edu.au
Ziqi Yan — z5506481@ad.unsw.edu.au
Sterling Song — z5513613@ad.unsw.edu.au

Contributions

Team members contributed to different parts of the project, including:

data preprocessing
baseline modelling
public AI detector evaluation
BERT-based detector evaluation
fine-tuning
model extension
result analysis
documentation