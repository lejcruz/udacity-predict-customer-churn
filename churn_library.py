# library doc string


# import libraries
import os
import logging
from typing import Callable

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import constants as cnts

os.environ['QT_QPA_PLATFORM']='offscreen'

logging.basicConfig(
    filename='./logs/results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Define constants based on contants.py
DATA_PATH = cnts.IN_DATA_PATH
EDA_PATH = cnts.OUT_EDA_PATH
TARGET_IN_COLUMN = cnts.TARGET_IN_COLUMN
TARGET_OUT_COLUMN = cnts.TARGET_OUT_COLUMN
TARGET_FUNCTION = cnts.TARGET_FUNCTION
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
        logging.info(message)  # Default to info if an invalid level is provided
    
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
def get_target(df: pd.DataFrame, column: str, lambda_function: Callable, target_name='target'):
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
        df[target_name] =  df[column].apply(lambda_function)
        df.drop(columns=column, inplace=True)
        log_message(f'SUCCESS - {target_name} target was defined based on {column}')
        return df
    
    except Exception as e:  
        log_message(f'ERROR - It was not possible to calculate the target column: {e}', level='error')
        return None

class Eda:
    
    def __init__(self, df: pd.DataFrame, out_pth: str, ids: list, response: str):
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
            log_message(f"The null percentage is {null_percent}", level='warning')
            return null_columns, null_percent
        
        else:
            log_message("No null values detected on df")
       
        return None, None
    

    def define_feature_types(self) -> tuple:
        """
        Separate quantitative and categorical features.
        """

        df_cols = [col for col in self.df.columns if col not in self.ids]
        quant_cols = self.df[df_cols].select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df[df_cols].select_dtypes(exclude=['number']).columns.tolist()

        return quant_cols, cat_cols
    

    def save_plot(self, plot, filename: str):
        """
        Save Seaborn plot as an image.
        """
        fig = plot.get_figure()
        fig.savefig(f"{self.out_pth}/{filename}", dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig)


    def eda_plots(self):
        """
        Generate and save EDA plots.
        """
        quant_cols, cat_cols = self.define_feature_types()

        # Quantitative plots
        for qcol in quant_cols:
            try:
                histplot = sns.histplot(self.df[qcol], stat='density', kde=True)
                self.save_plot(histplot, f"eda_quant_{qcol}.png")
                log_message(f"Saved EDA image for quantitative column: {qcol}")
            except Exception as e:
                log_message(f"ERROR - It's not possible to generate the plot for column {qcol}: {str(e)}", level='error')


        # Categorical plots
        for ccol in cat_cols:
            try:
                countplot = sns.countplot(x=ccol, data=self.df)
                self.save_plot(countplot, f"eda_categ_{ccol}.png")
                log_message(f"Saved EDA image for categorical column: {ccol}")
            except Exception as e:
                log_message(f"ERROR - It's not possible to generate the plot for column {ccol}: {str(e)}", level='error')

        # correlation plot
        try:
            plt.figure(figsize=(20,10)) 
            corrplot = sns.heatmap(self.df.corr(numeric_only=True), annot=False, cmap='Dark2_r', linewidths = 2)
            self.save_plot(corrplot, f"correlation_plot.png")
            log_message(f"Saved correlation plot image")

        except Exception as e:
            log_message(f"ERROR - It's not possible to generate the correlation plot: {str(e)}", level='error')
    

    def perform_eda(self):
        '''
        Perform EDA on the DataFrame and save figures.
        '''
        try:
            
            # check null values
            null_colums, null_percent = self.check_null()

            # describe dataset
            describe = self.df.describe()
            log_message("SUCCESS - Performed describe statiscs on Dataset", data=describe)

            # save EDA plots
            self.eda_plots()
            log_message("SUCCESS - EDA plots saved")

            null_colums, null_percent = self.check_null()
            return {'null_columns': null_colums,
                    'null_percent': null_percent,
                    'statistics': describe
                    }

        except Exception as e:
            log_message("ERROR - EDA could not be performed: {e}", level='error')
            return None


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass



if __name__ == "__main__":
    df = import_data(DATA_PATH)
    df = get_target(df, TARGET_IN_COLUMN, TARGET_FUNCTION, TARGET_OUT_COLUMN)

    eda = Eda(df, EDA_PATH, ['CLIENTNUM'], TARGET_OUT_COLUMN)
    eda.perform_eda()