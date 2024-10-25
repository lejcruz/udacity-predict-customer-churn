"""
Author: Lennon de Jesus Cruz
Date: October 25, 2024

Module for testing the `churn_library` using pytest.

This module contains fixtures and test functions to validate the functionality
of various classes and methods in the `churn_library` module, such as the EDA 
and Model classes. It also includes utility functions like data encoding and 
feature engineering.

Files:
    churn_library.py: Contains the core functionality being tested.
    constants.py: Stores constants such as paths used in the testing process.
    
Logging:
    All test logs are stored in `./logs/tests_churn_library.log`.

Tests:
    The following tests are included:
        - test_import: Tests data import functionality.
        - test_get_target: Verifies target column transformation.
        - test_check_null: Tests null value checks using the EDA class.
        - test_define_feature_types: Tests automatic feature type identification.
        - test_eda_plots: Ensures EDA plots are generated correctly.
        - test_encoder_helper: Validates the encoding of categorical features.
        - test_perform_feature_engineering: Ensures feature engineering works as expected.
        - test_model_train: Tests model training functionality.
        - test_model_predict: Validates model predictions.
        - test_model_report: Verifies the generation of model reports.
        - test_compare_models: Tests model comparison based on performance metrics.

Fixtures:
    - sample_df: Generates a sample DataFrame with null values for testing.
    - define_eda: Initializes the EDA class for testing.
    - sample_features_data: Creates sample data for training and testing.
    - logistic_model: Initializes a logistic regression model.
    - random_forest_model: Initializes a random forest model.
    - model_instance: Initializes a Model class instance for testing.

Usage:
    To run the tests, execute the following command:
    ```
    pytest churn_scripts_logging_and_tests.py
    ```
"""

import os
import logging
import pytest
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import constants
import churn_library as cls

logging.basicConfig(
    filename='./logs/tests_churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

OUT_TESTS_EDA_PATH = constants.OUT_TESTS_EDA_PATH
OUT_TESTS_MODEL_REPORTS_PATH = constants.OUT_TESTS_MODEL_REPORTS_PATH
OUT_TESTS_MODEL_ARTIFACTS_PATH = constants.OUT_TESTS_MODEL_ARTIFACTS_PATH


@pytest.fixture
def sample_df():
    """Fixture for creating a sample DataFrame with some null values."""
    data = {
        'A': [1, 2, None, 4],
        'B': [None, 2, 3, 4],
        'C': [1, 2, 3, 4],
        'D': ['1', '2', None, '4'],
        'E': ['A', 'B', 'A', 'B']
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def define_eda(sample_df):
    '''Fixture for creating EDA class to test the EDA functions'''
    eda = cls.Eda(df=sample_df,
                  out_pth=OUT_TESTS_EDA_PATH,
                  ids=[],
                  response='C')
    return eda


@pytest.fixture
def sample_features_data():
    """Fixture to create sample data for training and testing."""
    X_train, y_train = make_classification(
        n_samples=100, n_features=10, random_state=42)
    X_test, y_test = make_classification(
        n_samples=50, n_features=10, random_state=42)
    return pd.DataFrame(X_train), pd.Series(
        y_train), pd.DataFrame(X_test), pd.Series(y_test)


@pytest.fixture
def logistic_model():
    """Fixture to create a logistic regression model."""
    return LogisticRegression()


@pytest.fixture
def random_forest_model():
    """Fixture to create a random forest model."""
    return RandomForestClassifier()


@pytest.fixture
def model_instance():
    """Fixture to create a Model instance with temporary paths."""
    return cls.Model(
        experiment_name="test_experiment",
        out_image_pth=OUT_TESTS_MODEL_REPORTS_PATH,
        out_artifact_pth=OUT_TESTS_MODEL_ARTIFACTS_PATH)


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        cls.log_message(
            "Testing import_eda: The file wasn't found",
            level='error')
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        cls.log_message(
            "Testing import_data: The file doesn't appear to have rows and columns",
            level='error')
        raise err


def test_get_target(sample_df):
    '''Function for test the calculated result of the target colums'''
    df = cls.get_target(sample_df,
                        'C',
                        lambda val: 0 if val <= 2 else 1,
                        'new_target')

    # Check if the target is correctly calculated
    assert df['new_target'].tolist() == [0, 0, 1, 1]


def test_check_null(define_eda):
    '''Test function for the eda class and check_null function'''
    null_columns, null_percent = define_eda.check_null()

    # Check if the columns with null are identified
    assert null_columns == ['A', 'B', 'D']
    assert len(null_percent) == 3
    assert null_percent['A'] == 25.

    cls.log_message(
        "Testing check_null -  null_columns: {null_columns} ; null_percent: {null_percent}")


def test_define_feature_types(define_eda):
    '''Test function for the eda class and define_feature_types function'''
    quant_cols, cat_cols = define_eda.define_feature_types()

    # check if the quant and cat cols are being identified correctly
    assert 'A' in quant_cols
    assert 'D' in cat_cols
    assert len(quant_cols) == 3
    assert len(cat_cols) == 2

    cls.log_message(
        "Testing define_feature_types -  quant_cols: {quant_cols} ; cat_cols: {cat_cols}")


def test_eda_plots(define_eda):
    '''Test function for the eda class and test_eda_plots function'''
    # test the plot function
    define_eda.eda_plots()

    # Check if the plots were saved
    for col in ['A', 'B', 'C']:
        quant_plot_path = os.path.join(
            OUT_TESTS_EDA_PATH, f'eda_quant_{col}.png')
        assert os.path.exists(
            quant_plot_path), f"Quantitative plot for column '{col}' was not saved"

    # Check if correlation plot was saved
    corrplot_path = os.path.join(OUT_TESTS_EDA_PATH, 'correlation_plot.png')
    assert os.path.exists(corrplot_path), "Correlation plot was not saved"


def test_encoder_helper(sample_df):
    '''
    test encoder helper
    '''

    # perform encoder

    encoder = cls.encoder_helper(sample_df, ['E'], 'C')

    assert 'E_encoded' in encoder.columns
    assert encoder['E_encoded'].tolist() == [2.0, 3.0, 2.0, 3.0]

    # Ensure that original columns are dropped when keep_originals=False
    assert 'E' not in encoder.columns

    # Ensure that a KeyError is handled when a specified column is missing.
    key_err_encoder = cls.encoder_helper(sample_df, ['Z'], 'churn')
    assert key_err_encoder is None


def test_perform_feature_engineering(sample_df):
    '''
    test perform_feature_engineering
    '''

    X_train, _, y_train, _ = cls.perform_feature_engineering(sample_df, [
                                                                       'A'], 'C', 0.5)

    # test if splits where correctly made
    assert [len(X_train), len(y_train)] == [2, 2]

    # test if id columns is not in the final X set
    assert 'A' not in X_train.columns

# Test Model traning and report


def test_model_train(model_instance, logistic_model, sample_features_data):
    """Test if the model is trained successfully."""

    X_train, y_train, _, _ = sample_features_data
    trained_model = model_instance.model_train(
        logistic_model, X_train, y_train)

    assert trained_model is not None
    assert hasattr(model_instance, 'cls')


def test_model_predict(model_instance, logistic_model, sample_features_data):
    """Test if the model makes predictions."""
    X_train, y_train, X_test, _ = sample_features_data
    model_instance.model_train(logistic_model, X_train, y_train)
    y_train_preds, y_test_preds = model_instance.model_predict(X_test)

    assert len(y_train_preds) == len(y_train)
    assert len(y_test_preds) == len(X_test)


def test_model_report(model_instance, logistic_model, sample_features_data):
    """Test if the model generates a report without errors."""
    X_train, y_train, X_test, y_test = sample_features_data
    model_instance.model_train(logistic_model, X_train, y_train)
    model_instance.model_predict(X_test)
    model_instance.model_report(y_test)

    # Check if the report has been created successfully
    assert model_instance.report is not None


def test_compare_models(
        sample_features_data,
        model_instance,
        logistic_model,
        random_forest_model):
    """Test the compare_models function."""
    X_train, y_train, X_test, y_test = sample_features_data

    # Train two models
    model_instance.model_train(logistic_model, X_train, y_train)
    model_instance.model_predict(X_test)

    model2 = cls.Model(
        "test_model2",
        OUT_TESTS_MODEL_REPORTS_PATH,
        OUT_TESTS_MODEL_ARTIFACTS_PATH)
    model2.model_train(random_forest_model, X_train, y_train)
    model2.model_predict(X_test)

    # Compare the two models
    best_model = cls.compare_models([model_instance, model2], y_test)
    assert best_model in [
        model_instance.experiment_name,
        model2.experiment_name]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
