# udacity-predict-customer-churn

## Project Introduction

In this project, you will implement your learnings to identify credit card customers who are most likely to churn. The completed project will include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package can also be run interactively or from the command-line interface (CLI).

This project will give you practice using your skills for testing and logging and using the best coding practices from this lesson. It will also introduce you to a problem data scientists across companies always face. How do we identify (and later intervene with) customers likely to churn?

If you want to download the data and work locally with the data, start the Workspace and download the data from the data folder. This ensures all students have access to the dataset. Below, you have a sample of the dataset:

![image](https://github.com/user-attachments/assets/b12db1fe-af28-4ccd-b63a-5db95e91ce2f)

Note: This project provides all students with a Workspace with all the requirements to develop and run their project. If you want to run it locally, you can download all the files and use your own setup to develop the project.

# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project has the main goal to refactor the Customer Churn Model presented in the [churn_notebook.ipynb](churn_notebook.ipynb) using the best code practices and adding tests and logging to it. This project doesn't aim to improve the model accuracy.

The project includes:

* Data preprocessing;
* Exploratoty Data Analysis (EDA)
* Training and evaluating classification machine learning models 
* Reporting results with key metrics such as F1-score and ROC curves.
* Saving model artifacts and graphs images.
* Compare between a list of models candidates and return the best one

## Files and data description
Main Files:

* [churn_library.py](): This script contains the core classes and functions used for training, predicting, reporting, and comparing models.
* [churn_scripts_logging_and_tests.py](): This script includes unit tests for the methods and uses pytest for testing.
* [constants.py](): This script works as a model configuration file all paths and fixed values that will be used in the model run will be extract from this file
* [data/](): This directory contain the dataset for training and testing the model.
* [images/](): Stores generated output images, including ROC curves and classification reports.
* [models/](): Stores generated artifact outputs (saved as pickle files)
* [logs/](): Store the log files writen along the code execution
requirements_py3.9.txt: A list of dependencies required to run the project. Install them using pip install -r requirements_py3.9.txt.

#### Data

Below is a sample of the data.

![image](https://github.com/user-attachments/assets/b12db1fe-af28-4ccd-b63a-5db95e91ce2f)

* ID columns = [CLIENTNUM]
* Target Column = [Attrition_Flag] > 0 if "Existing Customer" else 1

## Running Files

#### Step 1: Setup Environment

Make sure you have Python 3.9 installed. Install the necessary dependencies by running:

```python
pip install -r requirements_py3.9.txt
```

#### Step 3: Model Config

All configurations are made in the [constants.py]() file, the model should run with the pre configurations in this file, but if you chance paths, name of files or want to change the random_state be sure to change in this file

#### Step 3: Running the tests

Before Run the model pipeline make sure its working fine, for that you can run the [churn_scripts_logging_and_tests.py]() file using both pytest or calling direct with python running:

```python
python churn_scripts_logging_and_tests.py
```
or 
```python
pytest churn_scripts_logging_and_tests.py
```

#### Step 3: Running churn_library.py

You can train and test the models by running the [churn_library.py](). The script will train the model, generate predictions, create a report, and save the model.

```python
python churn_library.py
```
This command will:

	1. Train a model with the data found in data/data/bank_data.csv.
	2. Evaluate the model on the test data (generated along the model pipeline)
	3. Generate reports including a classification report and ROC curve, saving them in the images/ and models/ folder.
	4. Generate the logs in the .logs/results.log


## Additional Information

Below are some evidences of tests and pylint score:

#### Tests Score - Using python churn_script_logging_and_tests.py
![image](/images/docs/tests_passed.png)

#### Pylint Scores
![image](/images/docs/pylint_churn_library.png)

![image](/images/docs/pylint_tests.png)
obs: was not possible to get a score > 7 on tests because of pystest fixtures