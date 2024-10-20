import os
import pytest
import logging
import churn_library as cls
import constants
import pandas as pd
import numpy as np

logging.basicConfig(
    filename='./logs/tests_churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

OUT_TESTS_EDA_PATH = constants.OUT_TESTS_EDA_PATH


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




def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = cls.import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		cls.log_message("Testing import_eda: The file wasn't found", level='error')
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		cls.log_message("Testing import_data: The file doesn't appear to have rows and columns", level='error')
		raise err
	
def test_get_target(sample_df):

	df = cls.get_target(sample_df,
					 'C',
					 lambda val: 0 if val <= 2 else 1,
					 'new_target')
	
    # Check if the target is correctly calculated
	assert df['new_target'].tolist() == [0, 0, 1, 1]

def test_check_null(define_eda):
	'''Test function for the eda class and check_null function'''
	null_columns, null_percent = define_eda.check_null()

    #Check if the columns with null are identified
	assert null_columns == ['A', 'B', 'D']
	assert len(null_percent) == 3
	assert null_percent['A'] == 25.

	cls.log_message("Testing check_null -  null_columns: {null_columns} ; null_percent: {null_percent}")

def test_define_feature_types(define_eda):
	'''Test function for the eda class and define_feature_types function'''
	quant_cols, cat_cols = define_eda.define_feature_types()

	#check if the quant and cat cols are being identified correctly
	assert 'A' in quant_cols
	assert 'D'in cat_cols
	assert len(quant_cols) == 3
	assert len(cat_cols) == 2

	cls.log_message("Testing define_feature_types -  quant_cols: {quant_cols} ; cat_cols: {cat_cols}")

def test_eda_plots(define_eda):

	#test the plot function
	define_eda.eda_plots()

 	# Check if the plots were saved
	for col in ['A', 'B', 'C']:
		quant_plot_path = os.path.join(OUT_TESTS_EDA_PATH, f'eda_quant_{col}.png')
		assert os.path.exists(quant_plot_path), f"Quantitative plot for column '{col}' was not saved"

	#Check if correlation plot was saved
	corrplot_path = os.path.join(OUT_TESTS_EDA_PATH, f'correlation_plot.png')
	assert os.path.exists(corrplot_path), f"Correlation plot was not saved"


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

	#Ensure that a KeyError is handled when a specified column is missing.
	key_err_encoder = cls.encoder_helper(sample_df, ['Z'], 'churn')
	assert key_err_encoder is None

def test_perform_feature_engineering(sample_df):
	'''
	test perform_feature_engineering
	'''

	X_train, X_test, y_train, y_test = cls.perform_feature_engineering(sample_df, ['A'], 'C', 0.5)

	# test if splits where correctly made
	assert [len(X_train), len(y_train)] == [2, 2]

	# test if id columns is not in the final X set
	assert 'A' not in X_train.columns


# def test_train_models(train_models):
# 	'''
# 	test train_models
# 	'''


if __name__ == "__main__":
	pass








