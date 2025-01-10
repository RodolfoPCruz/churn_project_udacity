# Churn

## Description

This project focuses on predicting customer churn using machine learning models. It produces insights through comprehensive exploratory data analysis (ADA). The workflow includes training and evaluating two distinct models to generate accurate churn predictions, providing a solution for customer retention strategies.

## Table of Contents

- Installation
- Files and Data Description
- Usage
- Features
- Testing

## Installation

Clone the repository and install dependencies: 

bash 

git clone https://github.com/RodolfoPCruz/churn_project_udacity.git

cd  churn_project_udacity 

pip install -r requirements.txt

## Files and Data Description

### Project Structure

The project is organized as follows:

![Screenshot from 2025-01-10 10-41-42](/home/rodolfo/Insync/rodolfopcruz2@gmail.com/Google Drive/Estudo/Udacity/Machine Learning DevOps Engineer - Nanodegree/Exercicios/Curso_2/Projeto_FINAL_CHURN/Screenshot from 2025-01-10 10-41-42.png)

## Usage

bash

python churn_library.py 

This command executes the entire machine learning workflow, starting from dataset loading to model evaluation, seamlessly running all steps in between.

bash

python -c "from churn_library import perform_eda,  import_data, FILE_PATH; perform_eda(import_data(FILE_PATH))"

This command executes only exploratory data analysis.

## Features

- Exploratory data analysis (EDA)

  It performs EDA and saves the plots of univariate analysis and bivariate analysis.

- Feature engineering

- Modeling

- Model Evaluation

  It saves a classification report for each model and the ROC curves.

- Feature Importance

## Testing

To ensure the project is working correctly, you can run the tests. The tests were implemented using the pytest framework. It is a dependacy that is specified in the requirements.txt file.

### Running Tests

To verify the project’s functionality, you can run the tests implemented using the pytest framework. Pytest is listed as a dependency in the requirements.txt file.