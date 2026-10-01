"""
===============================================================================
Time Series Project: Application for Forecasting Daily Electricity Consumption 
in Metropolitan France Excluding Corsica using AutoML Libraries
===============================================================================

This file is organised as follows:
1. Load the dataset
2. Feature Engineering
3. Machine Learning
   3.1 FLAML
   3.2 AutoGluon
"""
# Standard libraries
import random
import platform

# Other libraries
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import darts
import flaml


from darts import TimeSeries
from flaml import AutoML
from pickle import dump, HIGHEST_PROTOCOL
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Seaborn: {}'.format(sns.__version__))
print('Darts: {}'.format(darts.__version__))
print('FLAML: {}'.format(flaml.__version__))



# Constants
SEED = 0
MAX_ROWS_DISPLAY = 300
MAX_COLUMNS_DISPLAY = 150
PERIOD = 365
LAGS = 365
FOLDS = 10

# Set the random seed for reproducibility
random.seed(SEED)

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
split_ratio = 1 - PERIOD / len(raw_dataset)
split_index = int(split_ratio * len(raw_dataset))
train_dataset = raw_dataset.iloc[:split_index]
test_dataset = raw_dataset.iloc[split_index:]

# Display datasets information and description
dataset_info_description(dataset=train_dataset, max_rows=150)
dataset_info_description(dataset=test_dataset, max_rows=150)


# Save the train dataset in CSV format
OUTPUT_CSV = 'datasets/daily consumption/automl/train_dataset.csv'
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



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
print('\n\n\n3. Machine Learning')


# 3.1 FLAML
print(f'\n\n3.1 FLAML')

# Instantiate the model
flaml_automl = AutoML()
flaml_automl.fit(
    dataframe=train_dataset,
    label='Consumption (MW)',
    metric='auto',
    task='ts_forecast',
    n_jobs=-1,
    eval_method='auto',
    n_splits=FOLDS,
    seed=SEED,
    early_stop=True,
    period=len(test_dataset)
)

# Display information about the best model
print('\nBest estimator: {}'.format(flaml_automl.best_estimator))
print('Best hyperparameters:\n{}'.format(flaml_automl.best_config))
print('Best loss: {}'.format(flaml_automl.best_loss))
print('Training time: {}s'.format(flaml_automl.best_config_train_time))

# Generate forecasts
forecasts_dataset = flaml_automl.predict(len(test_dataset))
forecasts_dataset.name = 'Consumption (MW)'
forecasts_dataset = forecasts_dataset.to_frame()
forecasts_dataset['Date'] = test.time_index.tolist()

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
evaluate_regression(y_test=test.values(), y_pred=forecasts.values(), seed=SEED)

# Plot actual values
plt.figure(figsize=(12, 6))
train.plot(label='Train', color='tab:blue')
test.plot(label='Test', color='tab:red')
forecasts.plot(label='Forecasts', color='tab:green')
plt.title(label=f'Actual Values vs Forecasts')
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()

# Model persistence
model_path = 'models/daily consumption/flaml/model.pkl'
with open(model_path, 'wb') as f:
    dump(automl, f, HIGHEST_PROTOCOL)


# 3.2 AutoGluon
print(f'\n\n3.2 AutoGluon')

# Create a feature to comply with the AutoGluon library’s requirements and
# convert the train dataset into AutoGluon DataFrame
train_set = train_dataset.copy()
train_set['item_id'] = 'consumption'
train_set = TimeSeriesDataFrame.from_data_frame(
    train_set,
    id_column='item_id',
    timestamp_column='Date'
)

# Instantiate the model
autogluon_automl = TimeSeriesPredictor(
    target='Consumption (MW)',
    prediction_length=len(test_dataset),
    freq='D',
    eval_metric_seasonal_period=PERIOD,
    path='models/daily consumption/autogluon/model'
)

# Train the model
autogluon_automl.fit(
    train_data=train_set,
    presets='high_quality',
    enable_ensemble=True,
    random_seed=SEED
)

# Display the best model
print('\nThe best model:\n{}'.format(
    autogluon_automl.leaderboard(data=train_set, extra_info=True)))

# Generate forecasts
forecasts_result = autogluon_automl.predict(train_set)
forecasts_dataset = forecasts_result.reset_index()[['timestamp', 'mean']]
forecasts_dataset = forecasts_dataset.rename(
    columns={'timestamp': 'Date', 'mean': 'Consumption (MW)'})

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
evaluate_regression(y_test=test.values(), y_pred=forecasts.values(), seed=SEED)

# Plot actual values
plt.figure(figsize=(12, 6))
train.plot(label='Train', color='tab:blue')
test.plot(label='Test', color='tab:red')
forecasts.plot(label='Forecasts', color='tab:green')
plt.title(label=f'Actual Values vs Forecasts')
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()
