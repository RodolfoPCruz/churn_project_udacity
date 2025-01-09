# library doc string

"""
churn_library.py
----------------
----------------
A module to predict whether a customer will churn.

This module includes:

This module includes:
    -Reading csv files;
    -Exploratory data analysis ( creation of bar plots, histograms, and heatmaps);
    -Feature engineering;
    -Training of machine learning models;
    -Creation of classification reports and roc curves;
    -Calculation of feature importance;
    -The results of the exploratory data analysis and classification are saved in files.

Example:

    from churn_library import import_data
    from churn_library import perform_eda
    from churn_library import encoder_helper
    from churn_library import perform_feature_engineering
    from churn_library import train_models

    dataframe = import_data("./data/bank_data.csv")
    perform_eda(dataframe)
    categorical_features = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    dataframe = encoder_helper(dataframe, categorical_features)
    input_train, input_test, output_train, output_test = perform_feature_engineering(
        dataframe)
    train_models(input_train, input_test, output_train, output_test)

Author: Rodolfo Cruz
Date: 2025-01-07
"""


# import libraries
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import set_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

from constants import (CHURN_DISTRIBUTION_PLOT_PATH,
                       CUSTOMER_AGE_DISTRIBUTION_PLOT_PATH,
                       FEATURE_IMPORTANCES_PLOT_PATH, FILE_PATH,
                       HEATMAP_PLOT_PATH, LOGISTIC_REGRESSION_MODEL_PATH,
                       LOGISTIC_REGRESSION_RESULTS_PATH,
                       MARITAL_STATUS_DISTRIBUTION_PLOT_PATH, PARAM_REPONSE,
                       RANDOM_FOREST_MODEL_PATH, RANDOM_FOREST_RESULTS_PATH,
                       ROC_CURVE_RESULT_PATH,
                       TOTAL_TRANSACTION_DISTRIBUTION_PLOT_PATH)
from constants import CATEGORICAL_COLUMNS

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


set_config(transform_output="pandas")
sns.set_style("darkgrid")


def import_data(pth):
    """
    Returns dataframe for the csv found at pth

    Args:
            pth (str): a path to the csv
    Returns:
            df: pandas dataframe
    """

    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    """
    Perform exploratory data anlysis on df and save figures to images folder

    Args:
            df: pandas dataframe

    Returns:
            None
    """

    sns.histplot(df['Attrition_Flag'])
    plt.savefig(CHURN_DISTRIBUTION_PLOT_PATH)
    plt.close()

    sns.histplot(df['Customer_Age'], bins=10)
    plt.savefig(CUSTOMER_AGE_DISTRIBUTION_PLOT_PATH)
    plt.close()

    sns.barplot(df.Marital_Status.value_counts('normalize'))
    plt.savefig(MARITAL_STATUS_DISTRIBUTION_PLOT_PATH)
    plt.close()

    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(TOTAL_TRANSACTION_DISTRIBUTION_PLOT_PATH)
    plt.close()

    sns.heatmap(
        df.corr(
            numeric_only=True),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig(HEATMAP_PLOT_PATH, bbox_inches='tight')
    plt.close()


def encoder_helper(df, category_lst, response = PARAM_REPONSE):
    """
    Helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    Args:
            df: pandas dataframe
            category_lst (list): list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
                        naming variables or index y column]

    Returns:
            df: pandas dataframe with new columns for all features in category_lst
    """


    df[response] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    for feature in category_lst:
        features_groups = df.groupby(feature).mean(numeric_only=True)[response]
        df[feature + '_' + response] = df[feature].map(features_groups)

    return df


def perform_feature_engineering(df, response = PARAM_REPONSE):
    """
    Performs feature engineering in the input pandas dataframe

    Args:
              df: pandas dataframe
              response (str): string of response name [optional argument that could be used
              for naming variables or index y column]

    Returns:
              x_train (pandas dataframe): x training data
              x_test (pandas dataframe): x testing data
              y_train (pandas series): y training data
              y_test (pandas series): y testing data
    """


    n_rows = len(df)
    y = df[response]
    columns_to_be_dropped = [response]

    for feature in df.columns:
        if df[feature].nunique() == n_rows or df[feature].nunique() == 1:
            columns_to_be_dropped.append(feature)

        if not pd.api.types.is_numeric_dtype(df[feature]):
            columns_to_be_dropped.append(feature)

    df = df.drop(columns=columns_to_be_dropped)

    x_train, x_test, y_train, y_test = train_test_split(
        df, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    Produces classification report for training and testing results and stores report as image
    in images folder

    Args:
            y_train (pandas series): training response values
            y_test (pandas series):  test response values
            y_train_preds_lr (pandas series): training predictions from logistic regression
            y_train_preds_rf (pandas series): training predictions from random forest
            y_test_preds_lr (pandas series): test predictions from logistic regression
            y_test_preds_rf (pandas series): test predictions from random forest

    Returns:
             None
    """

    predictions = {
        'Randon Forest': {
            'train': y_train_preds_rf,
            'test': y_test_preds_rf,
            'file_path': RANDOM_FOREST_RESULTS_PATH},
        'Logistic Regression': {
            'train': y_train_preds_lr,
            'test': y_test_preds_lr,
            'file_path': LOGISTIC_REGRESSION_RESULTS_PATH}}
    for model_name in predictions:
        plt.rc('figure', figsize=(8, 8))
        # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old
        # approach
        plt.text(0.01, 1, str(model_name + ' Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.2, str(
                classification_report(
                    y_test, predictions[model_name]['test'])), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.5, str(model_name + ' Test'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    y_train, predictions[model_name]['train'])), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(predictions[model_name]['file_path'])
        plt.clf()


def roc_curve_plot(lr_model, rf_model, x_test, y_test):
    """
    Produces Roc for logistic regression and random forest models.
    The curve wll be created using the test data and the resulting plot
    will stored in the images/results/roc_curve_result.png path

    Args:
            lr_model (sklearn model object): logistic regrssion model object
            rf_model (sklearn model objectt): random forest model object
            x_test (pandas series): test input data
            y_test (pandas series):  test response values

    Returns:
        None
    """
    lrc_plot = RocCurveDisplay.from_estimator(lr_model, x_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(
        rf_model, x_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(ROC_CURVE_RESULT_PATH)
    plt.clf()


def feature_importance_plot(
        model,
        x_data,
        output_pth = FEATURE_IMPORTANCES_PLOT_PATH):
    """
    Creates and stores the feature importances in pth

    Args:
            model (sklearn model object): model object containing feature_importances_
            x_data (pandas dataframe): pandas dataframe of x values
            output_pth (str): path to store the figure

    Returns:
             None
    """
    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.clf()


def train_models(x_train, x_test, y_train, y_test):
    """
    Train, store model results: images + scores, and store models
    input:
              x_train (pandas dataframe): X training data
              x_test (pandas dataframe): X testing data
              y_train (pandas series): y training data
              y_test (pandas series): y testing data
    output:
              None
    """

    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']}
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    joblib.dump(cv_rfc.best_estimator_, RANDOM_FOREST_MODEL_PATH)
    joblib.dump(lrc, LOGISTIC_REGRESSION_MODEL_PATH)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    roc_curve_plot(lrc, cv_rfc.best_estimator_, x_test, y_test)
    feature_importance_plot(cv_rfc.best_estimator_, x_train)


if __name__ == "__main__":
    dataframe = import_data(FILE_PATH)
    perform_eda(dataframe)
    categorical_features = CATEGORICAL_COLUMNS
    dataframe = encoder_helper(dataframe, categorical_features)
    input_train, input_test, output_train, output_test = perform_feature_engineering(
        dataframe)
    train_models(input_train, input_test, output_train, output_test)
