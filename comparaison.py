import csv
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import os


def compare_csv(file1, file2):
    # Read the CSV files
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)

    # Ensure the CSV files have the same shape
    if data1.shape != data2.shape:
        raise ValueError("CSV files do not have the same shape")

    # Flatten the data to 1D arrays
    y_true = data1.values.flatten()
    y_pred = data2.values.flatten()

    # Calculate F1 score and accuracy
    f1 = f1_score(y_true, y_pred, average="weighted")
    accuracy = accuracy_score(y_true, y_pred)

    return f1, accuracy


# Example usage
cwd = os.getcwd()
file1 = cwd + "/ML-proj1/compare.csv"
file2 = cwd + "/ML-proj1/y_pred_Test.csv"
f1, accuracy = compare_csv(file1, file2)
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
