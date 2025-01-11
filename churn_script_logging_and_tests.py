"""
churn_script_logging_and_tests.py
----------------
----------------
Test the functions from the churn library.py

The following functions will be tested:
	- import_data;
    - perform_eda;
	- encoder_helper;
	- perform_feature_engineering;
	- train_models.

The logs will be stores in ./logs/churn_library.log

Author: Rodolfo Cruz
Date: 2025-01-07
"""


import glob
import logging

import pandas as pd
import pytest

from churn_library import (encoder_helper, import_data, perform_eda,
                           perform_feature_engineering, train_models)
from constants import (CATEGORICAL_COLUMNS, FEATURE_IMPORTANCES_PLOT_PATH,
                       FILE_PATH, LOGISTIC_REGRESSION_RESULTS_PATH, LOGS_PATH,
                       PARAM_REPONSE, RANDOM_FOREST_RESULTS_PATH,
                       ROC_CURVE_RESULT_PATH)

logging.basicConfig(
    filename = LOGS_PATH,
    level = logging.INFO,
    force = True,
    filemode = 'w',
    format = '%(name)s - %(levelname)s - %(message)s')


@pytest.fixture
def param_file_path():
    """
    Provides the path to load the data

    Returns:
                str: path of the file
    """
    return FILE_PATH


@pytest.fixture
def param_response():
    """
    Provides the name of the column containing the expected output (churn or not)

    Returns:
                str: name of the column containing the expected output
    """
    return PARAM_REPONSE


@pytest.fixture
def param_categorical_columns():
    """
    Provides the names of the categorical columns

    Returns:
            list: list containing all the categorical columns
    """
    return CATEGORICAL_COLUMNS


@pytest.fixture
def param_expected_results():
    """
    Provides the names of the files containing the results that will be saved

    Returns:
                list: list containing the names of the files
    """

    return [
        ROC_CURVE_RESULT_PATH.split('/')[-1],
        RANDOM_FOREST_RESULTS_PATH.split('/')[-1],
        LOGISTIC_REGRESSION_RESULTS_PATH.split('/')[-1],
        FEATURE_IMPORTANCES_PLOT_PATH.split('/')[-1]]


def test_import(param_file_path):
    """
    Test import_data for loading a pandas dataframe from a csv file.
    Expects a pandas dataframe containing rows and columns to be loaded from a directory

    Raises FileNotFoundError if the file is not found at the specified path
    Asserts that the loaded dataframe has rows and column

    Args:
                param_file_path (fixture): Provides the path of the file
    """

    try:
        df = import_data(param_file_path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(param_file_path):
    """
    Test perform_eda for generating and saving five plots
    Asserts that five plots were saved in ./images/eda/'

    Args:
         param_file_path (fixture): Provides the path to load the input data

    """

    perform_eda(import_data(param_file_path))
    files_in_directory = glob.glob('./images/eda/*.png')
    count = len(files_in_directory)

    try:
        assert count == 5
        logging.info("Sucess: All plots were successfully saved")
    except AssertionError as err:
        logging.error("It appears that not all plots were saved")
        raise err


def test_encoder_helper(
        param_file_path,
        param_categorical_columns,
        param_response):
    """
        Test encoder_helper for encoding the categorical features
        Asserts that a new column is created for each categorical column

        Args:
                param_file_path (fixture): Provides the path to load the input data
        param_categorical_columns (fixture): Provides the columns to be encoded
                param_response (fixture): Provides the name of the column
                                          containing the expected output (churn or not)
        """

    df = import_data(param_file_path)
    df = encoder_helper(df, param_categorical_columns)

    correctly_encoded = 0
    for feature in param_categorical_columns:
        if feature + '_' + param_response in df.columns:
            logging.info(f'Sucess: feature {feature} was encoded')
            correctly_encoded += 1

    try:
        assert correctly_encoded == len(param_categorical_columns)
        logging.info('Sucess: All categorical features were encoded')
    except AssertionError as err:
        logging.error('Not all categorical features were encoded')
        raise err


def test_perform_feature_engineering(param_categorical_columns):
    """
    Test perform_feature_engineering for the creation of the training and testing datasets
    Asserts x_train and x_test have rows and columns
    Asserts y_train and y_test have rows
    Asserts x_train only contains numeric columns

        Args:
        param_categorical_columns (fixture): Provides the names of the categorical columns
    """

    df = import_data("./data/bank_data.csv")
    df = encoder_helper(df, param_categorical_columns)
    x_train, x_test, y_train, y_test = perform_feature_engineering(df)

    try:
        assert x_train.shape[0] > 0
        assert x_train.shape[1] > 0
        logging.info('Sucess: x_train was sucessfully created')
    except AssertionError as err:
        logging.error(
            "The x_train data does not appear to have rows and columns")
        raise err

    try:
        assert y_train.shape[0] > 0
        logging.info('Sucess: y_train was sucessfully created')
    except AssertionError as err:
        logging.error("The y_train data does not appear to have rows")
        raise err

    try:
        assert x_test.shape[0] > 0
        assert x_test.shape[1] > 0
        logging.info('Sucess: X_test was sucessfully created')
    except AssertionError as err:
        logging.error(
            "The X_test data does not appear to have rows and columns")
        raise err

    try:
        assert y_test.shape[0] > 0
        logging.info('Sucess: y_train was sucessfully created')
    except AssertionError as err:
        logging.error("The y_test data does not appear to have rows")
        raise err

    try:
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info(
            "Sucess: Train and test datasets were sucessfully created")
    except AssertionError as err:
        logging.error(
            'Number of rows of x is not equal to the number of rows of y')
        raise err

    non_numeric_columns = []
    for feature in x_train.columns:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            logging.info(f'The {feature} is not numerical')
            non_numeric_columns.append(feature)
    try:
        assert len(non_numeric_columns) == 0
        logging.info('Success: The dataset contains only numeric columns')
    except AssertionError as err:
        logging.error('There are non numerical columns in the dataset')
        raise err


def test_train_models(param_categorical_columns, param_expected_results):
    """
    Test train_model for saving the trained models and the results
    Asserts that two models were saved
    Asserts that the roc curve plot was saved
    Asserts that the classification reports were saved
    Asserts that the feature importances plot was saved

    Args:
                param_categorical_columns (fixture): Provides the names of the categorical columns
                param_expected_results (fixture): Provides the names of the files that were
                                                  expected to be saved
    """

    df = import_data("./data/bank_data.csv")
    df = encoder_helper(df, param_categorical_columns)
    x_train, x_test, y_train, y_test = perform_feature_engineering(df)
    train_models(x_train, x_test, y_train, y_test)

    count_saved_models = len(glob.glob('./models/*.pkl'))
    try:
        assert count_saved_models == 2
        logging.info('Sucess: The two models were sucessfully saved')
    except AssertionError as err:
        logging.error('Not all models were saved')
        raise err

    results_images = glob.glob('./images/results/*.png')
    results_images = [file_name.split('/')[-1] for file_name in results_images]

    try:
        assert len(results_images) == 4
        logging.info(
            'Sucess: The Roc curved was sucessfully created and saved')
        logging.info(
            'Sucess: The Logistic regression results and random forest results were saved')
        logging.info(
            'Sucess: The feature importances were calculated and saved')

    except AssertionError as err:
        for result in param_expected_results:
            if result not in results_images:
                logging.error(f'{result} was not saved')
        raise err


if __name__ == "__main__":
    pytest.main(["-x", "churn_script_logging_and_tests.py"])
