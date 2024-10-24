# Project 1 (CS-433): Predicting coronary heart disease 

This project is part of the Machine Learning Course (CS-433). Our objective is to predict the risk of heart attacks using data from over 300,000 individuals, leveraging binary classification techniques. The dataset is derived from the Behavioral Risk Factor Surveillance System (BRFSS) and includes various personal lifestyle and clinical features.



## Getting Started

To get started with this project, follow the steps below:

1. Clone the repository to your local machine.
2. Install the required dependencies.
3. Load the dataset and place it in the designated folder.
4. Run the main script ```run.py``` to execute the project.

## Dependencies

To run this project, you will need to install the following dependencies:

- Python 3.7 (or higher)
- NumPy
- Matplotlib

You can install these dependencies by running the following command:

```
pip install numpy matplotlib
```

## Dataset
The dataset used in this project is divided into : 
- ```x_train.csv```, ```y_train.csv``` : for training the model
- ```x_test.csv``` : to evaluate the model and generate submission predictions.

These files should be in the ```dataset_to_release``` folder, located on the same directory as the code when running the project.


## Project Structure

This repository contains the following main components:

```implementations.py```: This script contains the implementation of the machine learning methods required for the project. 

```run.py```: This script produces the final predictions in the format required for submission to the competition platform. It processes the data, trains the model, and generates the CSV file for submission.


```helpers.py```: Helper functions for loading data, creating CSV submissions, and basic utilities used throughout the project.

## Methods implemented

The following machine learning techniques have been implemented from scratch (as required):

- Linear Regression (Gradient Descent and Stochastic Gradient Descent)
- Least Squares Regression
- Ridge Regression
- Logistic Regression
- Regularized Logistic Regression

Each function returns the final weight vector and the corresponding loss value.

## Results

We evaluated the model using cross-validation and submitted our predictions to the competition platform for feedback. 

The final predictions are stored in a CSV file ```final_pred.csv``` for submission. 

## How to use 

1. Ensure the dataset is placed in the ```dataset_to_release``` folder.
2. Run the following command to execute the main script and generate predictions:

```bash
python run.py
```

3. The output will be saved as a CSV file in the current directory.



## Contributors

[@maenguye](https://github.com/maenguye)
[@CusumanoLeo](https://github.com/Cusumano) [@melinacherchali](https://github.com/melinacherchali) 
