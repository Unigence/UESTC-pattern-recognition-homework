# Pattern Recognition and Machine Learning Assignments

## Assignment 2: BP Neural Network and SVM

[`mlp_svm_gender.ipynb`](mlp_svm_gender.ipynb) contains the complete, executed second assignment in Chinese. It uses height, weight, shoe size, and 50 m time to compare a manually implemented **4–3–1 BP network** with linear, polynomial, RBF, and sigmoid SVMs.

All models share the same stratified **five outer folds**. Median imputation and standardization are fitted within each training fold; a separate five-fold inner loop selects hyperparameters by AUC. The report compares MLP with an SVM procedure that also selects its kernel inside the inner loop. Male is the positive class: SE is male recall and SP is female recall. Tables report five-fold mean ± sample standard deviation for SE, SP, ACC, and AUC.

Install the pinned project dependencies and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab mlp_svm_gender.ipynb
```

To execute the entire notebook non-interactively using the current project interpreter:

```powershell
.\.venv\Scripts\python.exe tools/run_homework2_notebook.py
```

- [Word report](report/模式识别第二次作业-实验报告.docx), with editable MathType equations and three-line tables.
- [PDF report](report/模式识别第二次作业-实验报告.pdf), exported from Word for convenient reading.
- [Report source](report/homework2_report.md) and [build notes](report/README.md).
- [`results/homework2/`](results/homework2/): protocol, data audit, fold assignments, metrics, predictions, and figures.

## Assignment 1: Minimum Error Rate Bayesian Classifier

Using the two features of height (cm) and weight (kg), complete parameter estimation, probability density plotting, male-female classification, and interactive prediction.

## Viewing and Running

`bayes_height_weight.ipynb` contains the complete steps, formulas, charts, test results, and interactive input. All results are displayed within the Notebook.

Interactive input must be used within a running Notebook.

## Implementation Contents

1. Read the Excel file, exclude samples with illegal, invalid, or extreme values, resulting in a total of 1266 samples.
2. Split the training set and test set by category at approximately 80% / 20% (modifiable).
3. Plot the height and weight histograms for males and females separately, and calculate and display the one-dimensional normal probability density using maximum likelihood estimation and Bayesian estimation.
4. Use the two-dimensional mean and full covariance obtained by maximum likelihood estimation to construct a classifier, and classify according to the minimum error rate Bayesian decision rule.
5. Display the decision regions, test results, and interactive sample prediction.

## First-Time Environment Setup

Use an independent `.venv` environment; Python 3.10 is recommended. Execute in this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The data file `pattern_recognition_homework_dataset.xlsx` needs to be kept in the same directory as the Notebook.
