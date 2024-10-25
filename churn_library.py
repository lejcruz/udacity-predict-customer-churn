"""
Author: Lennon de Jesus Cruz
Date: October 25, 2024

Predict customer churn

In this project, I've implemented some class and functions to identify credit card customers 
who are most likely to churn. 
This is a Python package for a machine learning project that follows coding (PEP8) and 
engineering best practices for implementing software (modular, documented, and tested). 
The package can also be run interactively or from the command-line interface (CLI).

Functions:
----------
log_message(message: str, data=None, level: str = 'info'):
    Logs a message at a specified level with optional additional data.

import_data(pth: str):
    Loads a CSV file into a pandas DataFrame.

get_target(df: pd.DataFrame, column: str, lambda_function: Callable, target_name: str = 'target'):
    Transforms a specified column into a target variable using a provided function.

save_plot(plot, out_pth: str, filename: str):
    Saves a plot object to a specified directory with a given filename.

encoder_helper(df: pd.DataFrame, category_lst: list, target_column: str, 
               response: str = 'encoded', keep_originals: bool = False):
    Encodes categorical columns with values based on the target variable.

Classes:
--------
Eda:
    A class to perform EDA, generating plots for both quantitative and categorical data.

Model:
    A class to manage machine learning model training, prediction, evaluation, and saving.
    
"""


# import libraries
import os
import logging
import time
from typing import Union, Callable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
import joblib
import shap

import constants as cnts

sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Define constants based on contants.py
DATA_PATH = cnts.IN_DATA_PATH
EDA_PATH = cnts.OUT_EDA_PATH
ID_COLUMNS_LST = cnts.ID_COLUMNS_LST
TARGET_IN_COLUMN = cnts.TARGET_IN_COLUMN
TARGET_OUT_COLUMN = cnts.TARGET_OUT_COLUMN
TARGET_FUNCTION = cnts.TARGET_FUNCTION
TEST_SIZE = cnts.TEST_SIZE
RANDOM_STATE = cnts.RANDOM_STATE
OUT_MODEL_REPORTS_PATH = cnts.OUT_MODEL_REPORTS_PATH
OUT_MODEL_ARTIFACTS_PATH = cnts.OUT_MODEL_ARTIFACTS_PATH

# cat_columns = cnts.CAT_COLUMNS
# quant_columns = cnts.QUANT_COLUMNS


def log_message(message: str, data=None, level: str = 'info'):
    """
    Helper function for logging messages at different levels.

    Parameters:
    - message: The log message.
    - data: Optional additional data to log.
    - level: The logging level ('info', 'error', 'warning', etc.).
    """
    if level == 'info':
        logging.info(message)
    elif level == 'error':
        logging.error(message)
    elif level == 'warning':
        logging.warning(message)
    elif level == 'debug':
        logging.debug(message)
    else:
        # Default to info if an invalid level is provided
        logging.info(message)

    if data is not None:
        logging.info(data)


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe or None if errors
    '''

    try:
        df = pd.read_csv(pth, index_col=0)
        log_message('SUCCESS - Data has been loaded')
        return df

    except FileNotFoundError:
        log_message('ERROR - The provided path is incorrect', level='error')
        return None

# define target column


def get_target(
        df: pd.DataFrame,
        column: str,
        lambda_function: Callable,
        target_name='target'):
    '''
    Apply a transformation to a specified column in the DataFrame to create a target column,
    and then remove the original column.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    column : str
        The name of the column to transform into the target.
    lambda_function : function
        A lambda function or any callable that will be applied to the specified column.
    target_name : str, optional, default='target'
        The name of the new target column created from the transformation.

    Returns:
    --------
    tuple or None:
        - If successful: Updated DataFrame.
        - If an error occurs: None

    '''

    try:
        df[target_name] = df[column].apply(lambda_function)
        df.drop(columns=column, inplace=True)
        log_message(
            f'SUCCESS - {target_name} target was defined based on {column}')
        return df

    except Exception as e:
        log_message(
            f'ERROR - It was not possible to calculate the target column: {e}',
            level='error')
        return None


def save_plot(plot, out_pth, filename: str):
    try:
        # Check if the plot is a Seaborn plot with `get_figure()`
        if hasattr(plot, 'get_figure'):
            fig = plot.get_figure()
        else:
            # If plot doesn't have `get_figure()`, assume it is a Matplotlib
            # figure
            fig = plt.gcf()

        plt.tight_layout()
        fig.savefig(
            f"{out_pth}/{filename}",
            dpi=fig.dpi,
            bbox_inches='tight',
            pad_inches=0)
        plt.close(fig)
        log_message(f'SUCCESS - Plot saved as {filename} at {out_pth}')

    except Exception as e:
        log_message(f'ERROR - Could not save the plot: {e}', level='error')


class Eda:

    def __init__(
            self,
            df: pd.DataFrame,
            out_pth: str,
            ids: list,
            response: str):
        self.df = df
        self.out_pth = out_pth
        self.ids = ids
        self.response = response

    def check_null(self):
        """
        Check for missing values in the DataFrame.
        """

        null_check = self.df.isnull().sum()
        null_columns = null_check[null_check > 0].index.tolist()
        null_sum = null_check.sum()

        # calculate the null percentage

        if null_sum > 0:
            null_percent = (null_check[null_columns] / len(self.df)) * 100
            log_message(
                f"The null percentage is {null_percent}",
                level='warning')
            return null_columns, null_percent

        else:
            log_message("No null values detected on df")

        return None, None

    def define_feature_types(self) -> tuple:
        """
        Separate quantitative and categorical features.
        """

        df_cols = [col for col in self.df.columns if col not in self.ids]
        quant_cols = self.df[df_cols].select_dtypes(
            include=['number']).columns.tolist()
        cat_cols = self.df[df_cols].select_dtypes(
            exclude=['number']).columns.tolist()

        return quant_cols, cat_cols

    def eda_plots(self):
        """
        Generate and save EDA plots.
        """
        quant_cols, cat_cols = self.define_feature_types()

        self.quant_cols = quant_cols
        self.cat_cols = cat_cols

        # Quantitative plots
        for qcol in quant_cols:
            try:
                histplot = sns.histplot(
                    self.df[qcol], stat='density', kde=True)
                save_plot(histplot, self.out_pth, f"eda_quant_{qcol}.png")
                log_message(f"Saved EDA image for quantitative column: {qcol}")
            except Exception as e:
                log_message(
                    f"ERROR - It's not possible to generate the plot for column {qcol}: {str(e)}",
                    level='error')

        # Categorical plots
        for ccol in cat_cols:
            try:
                countplot = sns.countplot(x=ccol, data=self.df)
                save_plot(countplot, self.out_pth, f"eda_categ_{ccol}.png")
                log_message(f"Saved EDA image for categorical column: {ccol}")
            except Exception as e:
                log_message(
                    f"ERROR - It's not possible to generate the plot for column {ccol}: {str(e)}",
                    level='error')

        # correlation plot
        try:
            plt.figure(figsize=(20, 10))
            corrplot = sns.heatmap(
                self.df.corr(
                    numeric_only=True),
                annot=False,
                cmap='Dark2_r',
                linewidths=2)
            save_plot(corrplot, self.out_pth, f"correlation_plot.png")
            log_message(f"Saved correlation plot image")

        except Exception as e:
            log_message(
                f"ERROR - It's not possible to generate the correlation plot: {str(e)}",
                level='error')

    def perform_eda(self):
        '''
        Perform EDA on the DataFrame and save figures.
        '''
        try:

            # check null values
            null_colums, null_percent = self.check_null()

            # describe dataset
            describe = self.df.describe()
            log_message(
                "SUCCESS - Performed describe statiscs on Dataset",
                data=describe)

            # save EDA plots
            self.eda_plots()
            log_message("SUCCESS - EDA plots saved")

            null_colums, null_percent = self.check_null()
            return {'null_columns': null_colums,
                    'null_percent': null_percent,
                    'statistics': describe
                    }

        except Exception as e:
            log_message(
                "ERROR - EDA could not be performed: {e}",
                level='error')
            return None


def encoder_helper(
        df,
        category_lst,
        target_colum,
        response='encoded',
        keep_originals=False):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn (target_colum) for each category

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    category_lst : list
        List of categorical columns to be encoded.
    target_colum : str
        The name of the target column to calculate mean values for encoding.
    response : str, optional, default='encoded'
        Suffix to add to the encoded column names.
    keep_originals : bool, optional, default=False
        Whether to keep the original categorical columns or drop them.

    Returns:
    --------
    pd.DataFrame
        The modified DataFrame with encoded categorical columns.
    '''

    try:
        for col in category_lst:
            cat_mean = df.groupby(col)[target_colum].mean()
            encoded_col_name = f'{col}_{response}'
            df[encoded_col_name] = df[col].map(cat_mean)

        log_message("Ecoding done successfully - Sample Data:")
        log_message("", df.head(2))

        if not keep_originals:
            return df.drop(columns=category_lst)

        return df

    except KeyError as err:
        log_message(f"ERROR - Column error: {err}", level="error")
        return None

    except Exception as e:
        log_message(f"ERROR - Unexpected error: {e}", level="error")
        return None


def perform_feature_engineering(
        df: pd.DataFrame,
        id_columns: list,
        target_column: str,
        test_size: float = 0.3,
        random_state: int = 42):
    """
    Performs feature engineering by splitting the DataFrame into training and testing sets.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the features and target column.
    id_columns : list
        List of columns that should be excluded from the feature set (e.g., identifiers).
    target_column : str
        Name of the target column.
    test_size : float, optional (default=0.3)
        Proportion of the dataset to include in the test split.
    random_state : int, optional (default=42)
        Random seed for reproducibility in the train-test split.

    Returns:
    --------
    X_train : pd.DataFrame
        Training set features.
    X_test : pd.DataFrame
        Testing set features.
    y_train : pd.Series
        Training set target.
    y_test : pd.Series
        Testing set target.

    Raises:
    -------
    ValueError:
        If the 'id_columns' is not a list or `target_column` is missing from the DataFrame.

    Example:
    --------
    >>> X_train, X_test, y_train, y_test = perform_feature_engineering(df, ['CLIENTNUM'], 'churn')
    """

    try:
        # Validate and join Id Columns and Target in a unique list
        if id_columns is not None and not isinstance(id_columns, list):
            log_message(
                f"ERROR id_columns must be a list. Received {type(id_columns).__name__}",
                level='error')
            raise ValueError

        X_drop_columns = id_columns + \
            [target_column] if id_columns else [id_columns]

        # Define X and Y
        y = df[target_column]
        X = df.drop(columns=X_drop_columns)

        # Split between train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        log_message(
            "SUCCESS - Data has been splited - shapes",
            data=[
                X_train.shape,
                X_test.shape,
                y_train.shape,
                y_test.shape])

    except ValueError as err:
        log_message(f"ERROR - Validation Failed:  {err}", level='error')
        return None, None, None, None

    except KeyError as err:
        log_message(
            f"ERROR - Some columns are missing in df:  {err}",
            level='error')
        return None, None, None, None

    except Exception as e:
        log_message(f"ERROR - Unexpected error:  {e}", level='error')
        return None, None, None, None

    return X_train, X_test, y_train, y_test


class Model:
    """
    A class to manage machine learning models, including training, predictions,
    reporting, and feature importance plotting.

    Attributes:
    ----------
    experiment_name : str
        Name of the experiment for logging and saving plots.

    Methods:
    -------
    model_train(classifier, X_train, y_train, grid_params={}, cv=5):
        Trains a model with optional grid search for hyperparameter tuning.

    model_predict(X_test):
        Makes predictions for train and test sets using the trained model.

    model_report(y_test):
        Generates a classification report for train and test sets and saves the report.

    plot_roc_curve():
        Plots and saves the ROC curve for the model.

    feature_importance():
        Plots and saves feature importance using SHAP values or coefficients.

    model_save():
        Saves the trained model to disk.

    run_model_pipeline(classifier, X_train, X_test, y_train, y_test, grid_params={}, cv=None):
        Executes the full machine learning pipeline: training, prediction, 
        reporting, and saving the model.
    """

    def __init__(
            self,
            experiment_name: str,
            out_image_pth: str,
            out_artifact_pth: bytes):
        """
        Initializes the Model object with the experiment name.

        Parameters:
        ----------
        experiment_name : str
            The name of the experiment, used for logging and saving plots.
        out_image_pth : str
            The path to save the model report image
        out_artifact_pth : str
            The path to save the model artifact usually a pikle file
        """

        self.experiment_name = experiment_name
        self.out_image_pth = out_image_pth
        self.out_artifact_pth = out_artifact_pth

    def model_train(self,
                    classifier: ClassifierMixin,
                    X_train: Union[pd.DataFrame, pd.Series],
                    y_train: Union[pd.DataFrame, pd.Series],
                    grid_params: dict = {},
                    cv: int = 5
                    ):
        """
        Trains a classifier model, optionally using GridSearchCV for hyperparameter tuning.

        Parameters:
        ----------
        classifier : ClassifierMixin
            The machine learning classifier to train.
        X_train : pd.DataFrame or pd.Series
            The features for training.
        y_train : pd.DataFrame or pd.Series
            The target variable for training.
        grid_params : dict, optional
            The hyperparameter grid for grid search (default is {}).
        cv : int, optional
            The number of cross-validation folds (default is 5).

        Returns:
        -------
        Trained model or best estimator from grid search.
        """

        # attribute X_train and y_train for latter use
        self.X_train = X_train
        self.y_train = y_train

        try:
            cls = classifier

            if grid_params:
                grid_cv = GridSearchCV(
                    estimator=cls, param_grid=grid_params, cv=cv)
                grid_cv.fit(X_train, y_train)
                self.cls = grid_cv.best_estimator_

                log_message(
                    f"Successfully fitted  grid search classifier and found best estimator")
                return grid_cv.best_estimator_

            else:
                cls.fit(X_train, y_train)
                self.cls = cls
                log_message(f"Successfully fitted the classifier")
                return cls

        except Exception as e:
            log_message(f"ERROR: Model training failed - {e}", level='error')

    def model_predict(self, X_test: Union[pd.DataFrame, pd.Series]):
        """
        Generates predictions for the training and test sets using the trained model.

        Parameters:
        ----------
        X_test : pd.DataFrame or pd.Series
            The test set features.

        Returns:
        -------
        y_train_preds, y_test_preds : tuple of arrays
            Predictions for training and test sets.
        """

        self.X_test = X_test

        try:
            y_train_preds = self.cls.predict(self.X_train)
            y_test_preds = self.cls.predict(X_test)
            log_message(f"Successfully predicted the model")

            self.y_train_preds, self.y_test_preds = y_train_preds, y_test_preds

            return y_train_preds, y_test_preds

        except Exception as e:
            log_message(
                f"ERROR: was not possible to predict the model: {e}",
                level='error')
            self.y_train_preds, self.y_test_preds = None, None

            return None, None

    def model_report(self, y_test):
        """
        Generates and saves a classification report for the training and test sets.

        Parameters:
        ----------
        y_test : pd.DataFrame or pd.Series
            The true target values for the test set.
        """
        try:
            self.y_test = y_test

            plt.rc('figure', figsize=(5, 5))
            plt.text(0.01, 1.25, str(f'{self.experiment_name} Train'), {
                     'fontsize': 10}, fontproperties='monospace')
            plt.text(0.01, 0.05, str(classification_report(y_test, self.y_test_preds)), {
                     'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
            plt.text(0.01, 0.6, str(f'{self.experiment_name} Test'), {
                     'fontsize': 10}, fontproperties='monospace')
            plt.text(0.01, 0.7, str(classification_report(self.y_train, self.y_train_preds)), {
                     'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
            plt.axis('off')

            self.report = plt
            save_plot(
                plt,
                self.out_image_pth,
                f'{self.experiment_name}_report.png')

        except Exception as e:
            log_message(
                f"ERROR: Failed to generate model report - {e}",
                level='error')

    def plot_roc_curve(self):
        """
        Plots and saves the ROC curve for the trained model.
        """
        try:
            plot = RocCurveDisplay.from_estimator(
                self.cls, self.X_test, self.y_test)
            save_plot(
                plot,
                self.out_image_pth,
                f'{self.experiment_name}_roccurve.png')

        except Exception as e:
            log_message(
                f"ERROR: Failed to plot ROC curve - {e}",
                level='error')

    def feature_importance(self):
        """
        Plots and saves the feature importance of the trained model, using SHAP values for tree-based models.
        """
        try:

            if isinstance(self.cls, LogisticRegression):
                importances = self.cls.coef_
                names = self.cls.feature_names_in_
                df_imp = pd.DataFrame(
                    importances, columns=names, index=['Coef']).T
                plot_imp = df_imp.plot(kind='bar', figsize=(10, 4))

            else:
                explainer = shap.TreeExplainer(self.cls)
                shap_values = explainer.shap_values(self.X_test)
                plot_imp = shap.summary_plot(
                    shap_values, self.X_test, plot_type="bar")

            save_plot(
                plot_imp,
                self.out_image_pth,
                f'{self.experiment_name}_feature_importances.png')

        except Exception as e:
            log_message(
                f"ERROR: Failed to plot feature importance - {e}",
                level='error')

    def model_save(self):
        """
        Saves the trained model as a pickle file.
        """
        try:
            # save best model
            joblib.dump(
                self.cls,
                os.path.join(
                    self.out_artifact_pth,
                    f'{self.experiment_name}.pkl'))
            log_message(
                f"Model saved successfully as {self.experiment_name}.pkl")

        except Exception as e:
            log_message(
                f"ERROR: Failed to save the model - {e}",
                level='error')

    def run_model_pipeline(
            self,
            classifier,
            X_train,
            X_test,
            y_train,
            y_test,
            grid_params={},
            cv=None):
        """
        Executes the entire machine learning pipeline: training, predicting, reporting, and saving the model.

        Parameters:
        ----------
        classifier : ClassifierMixin
            The classifier model to train and evaluate.
        X_train : pd.DataFrame
            Training data features.
        X_test : pd.DataFrame
            Test data features.
        y_train : pd.Series
            Training data target.
        y_test : pd.Series
            Test data target.
        grid_params : dict, optional
            Hyperparameters for grid search (default is {}).
        cv : int, optional
            Number of cross-validation folds (default is None).
        """

        start_time = time.time()  # Start the timer

        self.model_train(classifier, X_train, y_train, grid_params, cv)
        self.model_predict(X_test)
        self.model_report(y_test)
        self.plot_roc_curve()
        self.feature_importance()
        self.model_save()

        # Calculate and log elapsed time
        elapsed_time = time.time() - start_time  # End the timer
        log_message(
            f"Successfully runned the model pipeline and saved images and artifacts")
        log_message(f"Pipeline completed in {elapsed_time:.2f} seconds")


def compare_models(models: list, y_test: pd.Series, metric: str = 'f1-score'):
    """
    Compares multiple trained models and decides the best one based on the F1-score
    (or another metric if needed) from the classification reports.

    Parameters:
    ----------
    models : list
        A list of Model objects to compare.
    y_test : pd.Series
        The true labels for the test set.
    metric:
        The metric name in which to compare the models, must e compatible with the model_report

    Returns:
    -------
    str
        The name of the best model.
    """
    results = {}

    try:
        for model in models:
            # Get predictions for the current model
            y_test_preds = model.y_test_preds

            report = classification_report(
                y_test, y_test_preds, output_dict=True)
            metric_result = report['weighted avg'][metric]
            log_message(
                f"{model.experiment_name} {metric} (weighted avg): {metric_result:.4f}")

            # Store the score in results
            results[model.experiment_name] = metric_result

        # Determine the best model based on metric
        best_model_name = max(results, key=results.get)
        best_metric = results[best_model_name]

        log_message(
            f"The best model is {best_model_name} with {metric}: {best_metric:.4f}")

    except Exception as e:
        log_message(f"ERROR - Unexpected Error {e}", level='error')

    return best_model_name


if __name__ == "__main__":

    # Import data
    df = import_data(DATA_PATH)

    # Calcuted the target column (y)
    df = get_target(df, TARGET_IN_COLUMN, TARGET_FUNCTION, TARGET_OUT_COLUMN)

    # Perform EDA
    eda = Eda(df, EDA_PATH, ID_COLUMNS_LST, TARGET_OUT_COLUMN)
    eda.perform_eda()

    # Perform encoding in the categorical columns
    df_encoded = encoder_helper(df, eda.cat_cols, TARGET_OUT_COLUMN)

    # Split between train and test sets
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df_encoded, ID_COLUMNS_LST, TARGET_OUT_COLUMN, TEST_SIZE, RANDOM_STATE)

    # Run the model pipeline for logistic regression
    exp_logreg = Model("Logistic Regression",
                       OUT_MODEL_REPORTS_PATH,
                       OUT_MODEL_ARTIFACTS_PATH)

    exp_logreg.run_model_pipeline(
        LogisticRegression(
            solver='newton-cg',
            max_iter=3000),
        X_train,
        X_test,
        y_train,
        y_test)

    # Run the model pipeline for Random Forest
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    exp_random = Model("Random Forest",
                       OUT_MODEL_REPORTS_PATH,
                       OUT_MODEL_ARTIFACTS_PATH)

    exp_random.run_model_pipeline(RandomForestClassifier(random_state=42),
                                  X_train,
                                  X_test,
                                  y_train,
                                  y_test,
                                  param_grid,
                                  5)

    # compare the models
    compare_models([exp_logreg, exp_random], y_test)
