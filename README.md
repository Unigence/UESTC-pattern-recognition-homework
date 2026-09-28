# Pattern Recognition Assignment: Design of a Minimum Error Rate Bayesian Classifier Based on Height and Weight

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