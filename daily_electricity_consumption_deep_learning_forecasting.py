"""
===============================================================================
Time Series Project: Application for Forecasting Daily Electricity Consumption 
in Metropolitan France Excluding Corsica using NeuralProphet and
NeuralForecast Libraries
===============================================================================

This file is organised as follows:
1. Load the dataset
2. Feature Engineering
3. Machine Learning
   3.1 Functions
   3.2 NeuralProphet
   3.3 NeuralForecast
       3.3.1 AutoNHITS Estimator
       3.3.2 AutoLSTM Estimator
"""
# Standard libraries
import random
import platform

# Other libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import darts
import optuna
import neuralprophet
import neuralforecast


from darts import TimeSeries
from darts.metrics import mase
from optuna import create_study
from neuralprophet import NeuralProphet, set_random_seed, save
from neuralforecast.auto import AutoNHITS, AutoLSTM
from neuralforecast.core import NeuralForecast
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Seaborn: {}'.format(sns.__version__))
print('Darts: {}'.format(darts.__version__))
print('Optuna: {}'.format(optuna.__version__))
print('NeuralProphet: {}'.format(neuralprophet.__version__))
print('NeuralForecast: {}'.format(neuralforecast.__version__))



# Constants
SEED = 1
MAX_ROWS_DISPLAY = 300
MAX_COLUMNS_DISPLAY = 150
PERIOD = 365
LAGS = 365
FORECAST_HORIZON = 7

# Set the random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Set the maximum number of rows to display by Pandas
pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)

# Set the default Seaborn style
sns.set_style('whitegrid')



"""
===============================================================================
1. Load the dataset
===============================================================================
"""
print('\n\n\n1. Load the dataset')


# Load the dataset
INPUT_CSV = 'datasets/consommation-nationale-quotidienne.csv'
raw_dataset = load_dataset(file_path=INPUT_CSV, encoding='utf-8')



"""
===============================================================================
2. Feature Engineering
===============================================================================
"""
print('\n\n\n2. Feature Engineering')


# Split the dataset into train and test sets
train_dataset = raw_dataset.iloc[:-FORECAST_HORIZON]
test_dataset = raw_dataset.iloc[-FORECAST_HORIZON:]

# Display datasets information and description
dataset_info_description(dataset=train_dataset, max_rows=150)
dataset_info_description(dataset=test_dataset, max_rows=5)


# Save the train dataset in CSV format
OUTPUT_CSV = 'datasets/daily consumption/deepl learning/train_dataset.csv'
train_dataset.to_csv(OUTPUT_CSV, index=False)


# Create Darts train and test series from the Pandas DataFrames
train = TimeSeries.from_dataframe(
    df=train_dataset,
    time_col='Date',
    value_cols=['Consumption (MW)']
)
test = TimeSeries.from_dataframe(
    df=test_dataset,
    time_col='Date',
    value_cols=['Consumption (MW)']
)


# Rename the features to comply with the NeuralForecast library’s requirements
train_dataset = train_dataset.rename(columns={
    'Date': 'ds', 'Consumption (MW)': 'y'})
train_dataset['ds'] = pd.to_datetime(train_dataset['ds'])
train_dataset['unique_id'] = 'consommation'
train_dataset = train_dataset[['unique_id', 'ds', 'y']]
test_dataset = test_dataset.rename(columns={
    'Date': 'ds', 'Consumption (MW)': 'y'})
test_dataset['ds'] = pd.to_datetime(test_dataset['ds'])
test_dataset['unique_id'] = 'consommation'
test_dataset = test_dataset[['unique_id', 'ds', 'y']]



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
print('\n\n\n3. Machine Learning')


# 3.1 Functions
print(f'\n\n3.1 Functions')

def neuralprophet_model_optimisation(trial):
    """This function generates a hyperparameter configuration for 
    the NeuralProphet model using Optuna, trains the model, generates 
    forecasts, evaluates performance, and returns a score for optimisation.

    Args:
        trial (optuna.Trial): an Optuna trial object, used to suggest
                              hyperparameter values

    Returns:
        score (float): the result of model evaluation
    """
    
    # Split the train dataset into train and validation sets
    X_train_opt = X_train.iloc[:-FORECAST_HORIZON]
    X_val_opt = X_train.iloc[-FORECAST_HORIZON:]
    
    # Instantiate the model
    growth = trial.suggest_categorical(
        'growth', ['off', 'linear', 'discontinuous'])
    n_changepoints = trial.suggest_int(
        name='n_changepoints', low=1, high=500, log=False)
    changepoints_range = trial.suggest_float(
        name='changepoints_range', low=0, high=1, log=False)
    trend_reg=trial.suggest_float(
        name='trend_reg', low=0, high=500, log=False)
    trend_reg_threshold=trial.suggest_categorical(
        'trend_reg_threshold', [True, False])
    seasonality_mode = trial.suggest_categorical(
            'seasonality_mode', ['additive', 'multiplicative'])
    seasonality_reg = trial.suggest_float(
        name='seasonality_reg', low=0, high=500, log=False)
    future_regressors_model = trial.suggest_categorical(
        'future_regressors_model', [
            'neural_nets', 'shared_neural_nets', 'shared_neural_nets_coef'
        ]
    )
    future_reg_d_hidden = trial.suggest_categorical(
        'future_regressors_d_hidden', [64, 128, 256, 512])
    future_reg_num_hidden_layers = trial.suggest_int(
        name='future_regressors_num_hidden_layers', low=1, high=4, log=False)
    learning_rate = trial.suggest_float(
        name='learning_rate', low=1e-4, high=1e-1, log=False)
    epochs = trial.suggest_int(name='epochs', low=50, high=500, log=False)
    model = NeuralProphet(
        growth=growth,
        n_changepoints=n_changepoints,
        changepoints_range=changepoints_range,
        trend_reg=trend_reg,
        trend_reg_threshold=trend_reg_threshold,
        seasonality_mode=seasonality_mode,
        seasonality_reg=seasonality_reg,
        future_regressors_model=future_regressors_model,
        future_regressors_d_hidden=future_reg_d_hidden,
        future_regressors_num_hidden_layers=future_reg_num_hidden_layers,
        learning_rate=learning_rate,
        epochs=epochs
    )
    model = model.add_country_holidays(country_name='FR')
    
    # Train the model
    model.fit(df=X_train_opt, freq='D')
    
    # Generate forecasts
    forecasts_opt = model.predict(df=X_val_opt)
    forecasts_result_opt = forecasts_opt[['ds', 'yhat1']]
    
    # Create Darts series from the Pandas DataFrames
    train_opt = TimeSeries.from_dataframe(
        df=X_train_opt.reset_index(),
        time_col='ds',
        value_cols=['y']
    )
    val_opt = TimeSeries.from_dataframe(
        df=X_val_opt.reset_index(),
        time_col='ds',
        value_cols=['y']
    )
    forecasts_opt = TimeSeries.from_dataframe(
        df=forecasts_result_opt,
        time_col='ds',
        value_cols=['yhat1']
    )
    
    # Evaluation
    score = mase(
        actual_series=val_opt,
        pred_series=forecasts_opt,
        insample=train_opt,
        m=PERIOD
    )
    return score


def get_model_forecasts(estimator, estimator_name: str):    
    """This function trains a NeuralForecast estimator, generates forecasts, 
    evaluates performance, and plots the results.

    Args:
        estimator (neuralforecast.auto): the estimator to train and use for 
                                         forecasts
        estimator_name (str): the name of the estimator
    """
    
    print(f'\n\n{estimator_name}:')
    
    # Instantiate the model
    model = NeuralForecast(models=[estimator], freq='D')

    # Train the model
    model.fit(df=train_dataset)
    
    # Model path
    model_path = f'models/daily consumption/neuralforecast/{estimator_name}'
    
    # Model persistence
    model.save(path=model_path, overwrite=True)
    
    # Load the pre-trained model
    model = NeuralForecast.load(path=model_path)
    
    # Generate forecasts
    forecasts_result = model.predict(h=len(test_dataset))
    forecasts_dataset = forecasts_result.reset_index()[
        ['ds', f'{estimator_name}']]
    forecasts_dataset = forecasts_dataset.rename(
        columns={'ds': 'Date', f'{estimator_name}': 'Consumption (MW)'})

    # Create the Darts forecasts series from the Pandas DataFrame
    forecasts = TimeSeries.from_dataframe(
        df=forecasts_dataset,
        time_col='Date',
        value_cols=['Consumption (MW)']
    )

    # Evaluation
    print('\nEvaluation')
    evaluate_time_series(
        test=test, forecasts=forecasts, train=train, period=PERIOD)
    evaluate_regression(
        y_test=test.values(), y_pred=forecasts.values(), seed=SEED)
    
    # Plot actual values
    plt.figure(figsize=(12, 6))
    train.plot(label='Train', color='tab:blue')
    test.plot(label='Test', color='tab:red')
    forecasts.plot(label='Forecasts', color='tab:green')
    plt.title(label=f'Actual Values vs Forecasts')
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()


def autonhits_estimator_optimisation(trial):
    """This function generates a hyperparameter configuration for 
    the AutoNHITS estimator using Optuna.

    Args:
        trial (optuna.Trial): an Optuna trial object, used to suggest
                              hyperparameter values

    Returns:
        dict: a dictionary containing AutoNHITS hyperparameters for
              optimisation
    """
    
    return {
        'input_size': trial.suggest_categorical(
            'input_size', (365, 2 * 365, 3 * 365)),
        'n_blocks': trial.suggest_categorical(
            'n_blocks', (3 * [2], 3 * [3], 3 * [4], 3 * [5], 3 * [6])),
        'mlp_units': trial.suggest_categorical(
            'mlp_units', (3 * [[256, 256]], 3 * [[512, 512]])),
        'n_pool_kernel_size': trial.suggest_categorical(
            'n_pool_kernel_size', ([30, 7, 1], [365, 30, 7, 1])),
        'n_freq_downsample': trial.suggest_categorical(
            'n_freq_downsample', ([30, 7, 1], [365, 30, 7, 1])),
        'max_steps': trial.suggest_int(
            name='max_steps', low=100, high=500, log=False),        
        'learning_rate': trial.suggest_float(
            name='learning_rate', low=1e-4, high=1e-1, log=False),
        'batch_size': 32,
        'windows_batch_size': trial.suggest_int(
            name='max_steps', low=256, high=1024, log=False),
        'scaler_type': trial.suggest_categorical(
            'scaler_type', ('standard', 'minmax', 'robust'))
    }


def autolstm_estimator_optimisation(trial):
    """This function generates a hyperparameter configuration for 
    the AutoLSTM estimator using Optuna.

    Args:
        trial (optuna.Trial): an Optuna trial object, used to suggest
                              hyperparameter values

    Returns:
        dict: a dictionary containing AutoLSTM hyperparameters for
              optimisation
    """
    
    return {
        'input_size': trial.suggest_categorical(
            'input_size', (365, 2 * 365, 3 * 365)),
        'encoder_n_layers': trial.suggest_int(
            name='encoder_n_layers', low=2, high=6, log=False),
        'encoder_hidden_size': trial.suggest_int(
            name='encoder_hidden_size', low=128, high=1024, log=False),
        'decoder_hidden_size': trial.suggest_int(
            name='decoder_hidden_size', low=128, high=1024, log=False),
        'batch_size': 32,
        'max_steps': trial.suggest_int(
            name='max_steps', low=10, high=300, log=False),
        'learning_rate': trial.suggest_float(
            name='learning_rate', low=1e-4, high=1e-1, log=False),
        'scaler_type': trial.suggest_categorical(
            'scaler_type', ('standard', 'minmax', 'robust'))
    }


# 3.2 NeuralProphet
print(f'\n\n3.2 NeuralProphet')

try:
    # Set the random state for reproducibility
    set_random_seed(seed=SEED)
    
    # Drop the feature 'unique_id' to comply with the NeuralProphet
    # library’s requirements
    X_train = train_dataset.drop(['unique_id'], axis=1)
    X_test = test_dataset.drop(['unique_id'], axis=1)
    
    # Optimisation of the model
    callback = StopOptimisationEarlyCallback(stagnation_threshold=5)
    study = create_study(direction='minimize')
    study.optimize(
        func=neuralprophet_model_optimisation,
        n_jobs=-1,
        callbacks=[callback]
    )
    print(f'\nBest hyperparams: {study.best_params}')
    
    # Instantiate the model
    hyperparams = {
        'growth': study.best_params['growth'],
        'n_changepoints': study.best_params['n_changepoints'],
        'changepoints_range': study.best_params['changepoints_range'],
        'trend_reg': study.best_params['trend_reg'],
        'trend_reg_threshold': study.best_params['trend_reg_threshold'],
        'seasonality_mode': study.best_params['seasonality_mode'],
        'seasonality_reg': study.best_params['seasonality_reg'],
        'future_regressors_model': study.best_params[
            'future_regressors_model'],
        'future_regressors_d_hidden': study.best_params[
            'future_regressors_d_hidden'],
        'future_regressors_num_hidden_layers': study.best_params[
            'future_regressors_num_hidden_layers'],
        'learning_rate': study.best_params['learning_rate'],
        'epochs': study.best_params['epochs']
    }
    model = NeuralProphet(**hyperparams)
    model = model.add_country_holidays(country_name='FR')
    
    # Train the model
    model.fit(df=X_train, freq='D')
    
    # Model persistence
    model_path = 'models/daily consumption/neuralprophet/model.np'
    save(forecaster=model, path=model_path)
    
    # Generate forecasts
    forecasts_result = model.predict(df=X_test)
    forecasts_dataset = forecasts_result[['ds', 'yhat1']]
    forecasts_dataset = forecasts_dataset.rename(columns={
        'ds': 'Date', 'yhat1': 'Consumption (MW)'})
    
    # Create the Darts forecasts series from the Pandas DataFrame
    forecasts = TimeSeries.from_dataframe(
        df=forecasts_dataset,
        time_col='Date',
        value_cols=['Consumption (MW)']
    )
    
    # Evaluation
    print('\nEvaluation')
    evaluate_time_series(
        test=test, forecasts=forecasts, train=train, period=PERIOD)
    evaluate_regression(
        y_test=test.values(), y_pred=forecasts.values(), seed=SEED)
    
    # Plot actual values
    plt.figure(figsize=(12, 6))
    train.plot(label='Train', color='tab:blue')
    test.plot(label='Test', color='tab:red')
    forecasts.plot(label='Forecasts', color='tab:green')
    plt.title(label='Actual Values vs Forecasts')
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()
except Exception as error:
    print(f'The following error occurred: {error}')


# 3.3 NeuralForecast
print(f'\n\n3.3 NeuralForecast')

# 3.3.1 AutoNHITS Estimator
print(f'\n3.3.1 AutoNHITS Estimator')

# Instantiate the estimator
try:
    estimator = AutoNHITS(
        h=FORECAST_HORIZON,
        config=autonhits_estimator_optimisation,
        backend='optuna'
    )
    get_model_forecasts(estimator=estimator, estimator_name='AutoNHITS')
except Exception as error:
    print(f'The following error occurred: {error}')


# 3.3.2 AutoLSTM Estimator
print(f'\n3.3.2 AutoLSTM Estimator')

# Instantiate the estimator
try:
    estimator = AutoLSTM(
        h=FORECAST_HORIZON,
        config=autolstm_estimator_optimisation,
        backend='optuna'
    )
    get_model_forecasts(estimator=estimator, estimator_name='AutoLSTM')
except Exception as error:
    print(f'The following error occurred: {error}')
