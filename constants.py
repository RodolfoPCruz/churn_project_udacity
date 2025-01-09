"""
constants.py
----------------
----------------
Constants that will be used in churn_library.py and churn_script_logging_and_tests

Author: Rodolfo Cruz
Date: 2025-01-07
"""


# path of the dataset
FILE_PATH = "./data/bank_data.csv"

# Paths where the pĺots created during exploratory data analysis will be stored
CHURN_DISTRIBUTION_PLOT_PATH = './images/eda/churn_distribution.png'
CUSTOMER_AGE_DISTRIBUTION_PLOT_PATH = './images/eda/customer_age_distribution.png'
MARITAL_STATUS_DISTRIBUTION_PLOT_PATH = './images/eda/marital_status_distribution.png'
TOTAL_TRANSACTION_DISTRIBUTION_PLOT_PATH = './images/eda/total_transaction_distribution.png'
HEATMAP_PLOT_PATH = './images/eda/heatmap.png'

# paths where the results obtained will be stored
RANDOM_FOREST_RESULTS_PATH = './images/results/rf_results.png'
LOGISTIC_REGRESSION_RESULTS_PATH = './images/results/lr_results.png'
ROC_CURVE_RESULT_PATH = './images/results/roc_curve_result.png'
FEATURE_IMPORTANCES_PLOT_PATH = './images/results/feature_importances.png'

# paths where the trained models will be stored
RANDOM_FOREST_MODEL_PATH = './models/rfc_model.pkl'
LOGISTIC_REGRESSION_MODEL_PATH = './models/logistic_model.pkl'

# paths where the logs will be stored
LOGS_PATH = './logs/churn_library.log'

# name of the column containing the expected output
PARAM_REPONSE = "Churn"

#name of the categorical columns in the dataset
CATEGORICAL_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']
