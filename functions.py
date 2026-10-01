"""
===============================================================================
This file contains all the functions for the project
===============================================================================
"""
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import optuna


from csv import Sniffer
from darts.metrics import mase, mse, mae, mape, smape, r2_score
from sklearn import metrics



def load_dataset(file_path, encoding):
    """This function loads a csv file and finds the type of separators by
     sniffing the file.

    Args:
        file_path (str): the csv file path
        encoding (str): encoding to use for reading and writing

    Returns:
        dataset (pd.DataFrame): the loaded Pandas data set
    """

    with open(file_path, 'r') as csvfile:
        separator = Sniffer().sniff(csvfile.readline()).delimiter
    dataset = pd.read_csv(
        filepath_or_buffer=file_path,
        sep=separator,
        encoding=encoding,
        encoding_errors='ignore',
        on_bad_lines='skip'
    )
    return dataset


def dataset_info_description(dataset, max_rows: int):
    """This function displays the information and description of a Pandas
    DataFrame.

    Args:
        dataset (pd.DataFrame): the Pandas DataFrame
        max_rows (int): the maximum number of rows in the dataset to be
                        displayed
    """

    # Display dimensions of the dataset
    print(f'\nDimensions of the dataset: {dataset.shape}')

    # Display information about the dataset
    print(f'\nInformation about the dataset:\n{dataset.info()}')

    # Display the description of the dataset
    print(f'\nDescription of the dataset:\n{dataset.describe(include="all")}')

    # Display the head and the tail of the dataset
    print(f'\nDescription of the dataset:\n'
          f'{pd.concat([dataset.head(max_rows), dataset.tail(max_rows)])}')


def evaluate_time_series(test, forecasts, train, period):
    """This function evaluates Time Series model performance 
    using Darts library.

    Args:
        test (darts.timeseries.TimeSeries): the test set series
        forecasts (darts.timeseries.TimeSeries): the series containing the
                                                 forecasts
        train (darts.timeseries.TimeSeries): the train set series for insample 
                                             to calculate the reference error
        PERIOD (int): the seasonal period of the Time Series data set
    """
    
    print('\nMASE: {:.3f}'.format(mase(
        actual_series=test, pred_series=forecasts, insample=train, m=period)))
    print('MSE: {:.3f}'.format(mse(
        actual_series=test, pred_series=forecasts)))
    print('MAE: {:.3f}'.format(mae(
        actual_series=test, pred_series=forecasts)))
    print('MAPE: {:.3f}'.format(mape(
        actual_series=test, pred_series=forecasts)))
    print('SMAPE: {:.3f}'.format(smape(
        actual_series=test, pred_series=forecasts)))
    print('R²: {:.3f}'.format(r2_score(
        actual_series=test, pred_series=forecasts)))


def evaluate_regression(y_test, y_pred, seed):
    """This function evaluates Regression model performance 
    using Scikit-learn library.

    Args:
        y_test (array-like): the test labels
        y_pred (array-like): the predicted labels
        seed (int): the random state value in order to ensure reproducibility
    """

    print('\nMSE: {:.3f}'.format(metrics.mean_squared_error(
        y_test, y_pred)))
    print('MAE: {:.3f}'.format(metrics.mean_absolute_error(
        y_test, y_pred)))
    print('MAPE: {:.3f}'.format(metrics.mean_absolute_percentage_error(
        y_test, y_pred)))
    print('MdAE: {:.3f}'.format(metrics.median_absolute_error(
        y_test, y_pred)))
    if np.where(y_test < 0)[0].size == 0 and np.where(y_pred < 0)[0].size == 0:
        print('MSLE: {:.3f}'.format(metrics.mean_squared_log_error(
            y_test, y_pred)))
    elif np.where(y_test < 0)[0].size > 0:
        print('Impossible to compute MSLE because the test set contains '
              'negative values.')
    elif np.where(y_pred < 0)[0].size > 0:
        print('Impossible to compute MSLE because forecasts or predictions '
              'contain negative values.')
    print('Maximum residual error: {:.3f}'.format(metrics.max_error(
        y_test, y_pred)))
    print('Explained variance score: {:.3f}'.format(
        metrics.explained_variance_score(y_test, y_pred)))
    print('R²: {:.3f}'.format(metrics.r2_score(y_test, y_pred)))

    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    metrics.PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred,
        kind='actual_vs_predicted',
        ax=axs[0],
        random_state=seed
    )
    axs[0].set_title('Actual Values vs Predicted Values')
    axs[0].grid(True)
    metrics.PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred,
        kind='residual_vs_predicted',
        ax=axs[1],
        random_state=seed
    )
    axs[1].set_title('Residuals Values vs Predicted Values')
    axs[1].grid(True)
    fig.suptitle('Plot the results of predictions')
    plt.show()


class StopOptimisationEarlyCallback:
    """A callback for Optuna that stops optimisation when there is no
    improvement after a set number of trials.

    Attributes:
        stagnation_threshold (int): the number of trials without improvement
                                    before stopping
        best_value (float): the best score achieved so far
        stagnation_counter (int): the counter of trials without improvement
    """

    def __init__(self, stagnation_threshold):
        self.stagnation_threshold = stagnation_threshold
        self.best_value = None
        self.stagnation_counter = 0


    def __call__(self, study: optuna.study.Study,
                 trial: optuna.trial.FrozenTrial):

        # If a new better score has been found
        if study.best_trial.number == trial.number:
            self.best_value = study.best_value
            self.stagnation_counter = 0
        else:
            # If best score is already defined
            if self.best_value is not None:
                self.stagnation_counter += 1

                # Stop if stagnation threshold is reached
                if self.stagnation_counter >= self.stagnation_threshold:
                    print(f'\nStopping after {trial.number} trials')
                    print(f'Last improvement at trial: '
                          f'{study.best_trial.number}')
                    print(f'Best score: {study.best_value}')
                    print(f'Best hyperparameters: {study.best_params}')
                    study.stop()
