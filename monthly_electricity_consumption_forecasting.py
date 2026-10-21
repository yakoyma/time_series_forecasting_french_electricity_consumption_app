"""
===============================================================================
Time Series Project: Application for Forecasting the Monthly Electricity
Consumption in Metropolitan France Excluding Corsica using Darts and PyCaret
Libraries
===============================================================================

This file is organised as follows:
1. Load the dataset
2. Feature Engineering
3. Machine Learning
   3.1 Darts
   3.2 PyCaret
"""
# Standard libraries
import random
import platform

# Other libraries
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
import darts
import joblib
import pycaret


from sklearn.ensemble import ExtraTreesRegressor
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import (NaiveDrift,
                          NaiveMean,
                          NaiveMovingAverage,
                          NaiveSeasonal,
                          GlobalNaiveAggregate,
                          Prophet,
                          AutoARIMA,
                          AutoTheta,
                          AutoTBATS,
                          AutoETS,
                          AutoCES,
                          AutoMFLES,
                          LinearRegressionModel,
                          RandomForestModel,
                          LightGBMModel,
                          XGBModel,
                          CatBoostModel,
                          SKLearnModel,
                          TSMixerModel,
                          DLinearModel,
                          NeuralForecastModel,
                          TCNModel,
                          NBEATSModel)
from darts.metrics import mase, mae, r2_score
from joblib import dump
from pycaret.tasks import *
from pycaret import save_model
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Seaborn: {}'.format(sns.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('Darts: {}'.format(darts.__version__))
print('Joblib: {}'.format(joblib.__version__))
print('PyCaret: {}'.format(pycaret.__version__))



# Constants
SEED = 0
MAX_ROWS_DISPLAY = 200
MAX_COLUMNS_DISPLAY = 50
PERIOD = 12
LAGS = 24
INPUT_CHUNK_LENGTH = 24
OUTPUT_CHUNK_LENGTH = 12

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
INPUT_CSV = 'datasets/consommation-nationale-mensuelle.csv'
raw_dataset = load_dataset(file_path=INPUT_CSV, encoding='utf-8')



"""
===============================================================================
2. Feature Engineering
===============================================================================
"""
print('\n\n\n2. Feature Engineering')


# Create the Darts series from the Pandas DataFrame
dataset = TimeSeries.from_dataframe(
    df=raw_dataset,
    time_col='Date',
    value_cols=['Consumption (GWh)']
)


# Split the dataset into train and test sets
split_ratio = 1 - PERIOD / len(dataset)
train, test = dataset.split_after(split_ratio)


# Save the train dataset in CSV format
OUTPUT_CSV_1 = 'datasets/monthly consumption/train_dataset.csv'
train_dataset = train.to_dataframe().reset_index()
train_dataset.to_csv(OUTPUT_CSV_1, index=False)


# Normalisation
scaler = Scaler()
train_scaled = scaler.fit_transform(series=train)
test_scaled = scaler.transform(series=test)

# Scaler persistence
dump(
    value=scaler,
    filename='models/scaler/monthly consumption/darts/scaler.joblib'
)



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
print('\n\n\n3. Machine Learning')


# 3.1 Darts
print('\n\n3.1 Darts')

def get_model_forecasts(model, model_name: str):
    """This function trains a Darts model, generates forecasts,
    evaluates performance, and plots the results.

    Args:
        model (darts.models.forecasting): the model to train and use for
                                          forecasts
        model_name (str): the name of the model
    """

    print(f'\n\n{model_name}:')

    # Train the model
    model.fit(series=train_scaled)

    # Generate forecasts
    forecasts_scaled = model.predict(n=len(test_scaled))

    # Evaluation for scaled values
    print('\nEvaluation for scaled values')
    evaluate_time_series(
        test=test_scaled,
        forecasts=forecasts_scaled,
        train=train_scaled,
        period=PERIOD
    )
    evaluate_regression(
        y_test=test_scaled.values(),
        y_pred=forecasts_scaled.values(),
        seed=SEED
    )

    # Plot scaled values
    plt.figure(figsize=(12, 6))
    train_scaled.plot(label='Train', color='tab:blue')
    test_scaled.plot(label='Test', color='tab:red')
    forecasts_scaled.plot(label='Forecasts', color='tab:green')
    plt.title(label=f'{model_name} - Scaled Values vs Forecasts')
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()

    # Reverse normalisation
    forecasts = scaler.inverse_transform(series=forecasts_scaled)

    # Evaluation for actual values
    print('\nEvaluation for actual values')
    evaluate_time_series(
        test=test, forecasts=forecasts, train=train, period=PERIOD)
    evaluate_regression(
        y_test=test.values(), y_pred=forecasts.values(), seed=SEED)
    mase_score = mase(
        actual_series=test, pred_series=forecasts, insample=train, m=PERIOD)
    mae_score = mae(
        actual_series=test, pred_series=forecasts)
    rsquare_score = r2_score(actual_series=test, pred_series=forecasts)

    # Model persistence
    model_path = f'models/monthly consumption/darts/{model_name}_model.pkl'
    model.save(path=model_path)

    # Plot actual values
    plt.figure(figsize=(12, 6))
    train.plot(label='Train', color='tab:blue')
    test.plot(label='Test', color='tab:red')
    forecasts.plot(label='Forecasts', color='tab:green')
    plt.title(label=f'{model_name} - Actual Values vs Forecasts')
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()

    return mase_score, mae_score, rsquare_score


# Instantiate the models
models_list = [
    NaiveDrift(),
    NaiveMean(),
    NaiveMovingAverage(),
    NaiveSeasonal(K=PERIOD),
    GlobalNaiveAggregate(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH
    ),
    Prophet(random_state=SEED),
    AutoARIMA(season_length=PERIOD, random_state=SEED),
    AutoTheta(season_length=PERIOD, random_state=SEED),
    AutoTBATS(season_length=PERIOD, random_state=SEED),
    AutoETS(season_length=PERIOD, random_state=SEED),
    AutoCES(season_length=PERIOD, random_state=SEED),
    AutoMFLES(
        test_size=int((1 - PERIOD / len(train)) * len(train)),
        season_length=PERIOD,
        random_state=SEED
    ),
    LinearRegressionModel(
        lags=LAGS,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        random_state=SEED
    ),
    RandomForestModel(
        lags=LAGS,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        random_state=SEED
    ),
    LightGBMModel(
        lags=LAGS,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        random_state=SEED
    ),
    XGBModel(
        lags=LAGS,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        random_state=SEED
    ),
    CatBoostModel(
        lags=LAGS,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        random_state=SEED
    ),
    SKLearnModel(
        model=ExtraTreesRegressor(),
        lags=LAGS,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        random_state=SEED
    ),
    TSMixerModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH
    ),
    DLinearModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH
    ),
    NeuralForecastModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        model='MLP'
    ),
    TCNModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH
    ),
    NBEATSModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        layer_widths=512,
        num_blocks=2
    )
]
models_names_list = [
    'NaiveDrift',
    'NaiveMean',
    'NaiveMovingAverage',
    'NaiveSeasonal',
    'GlobalNaiveAggregate',
    'Prophet',
    'AutoARIMA',
    'AutoTheta',
    'AutoTBATS',
    'AutoETS',
    'AutoCES',
    'AutoMFLES',
    'LinearRegressionModel',
    'RandomForestModel',
    'LightGBMModel',
    'XGBModel',
    'CatBoostModel',
    'SKLearnModel',
    'TSMixerModel',
    'DLinearModel',
    'NeuralForecastModel',
    'TCNModel',
    'NBEATSModel'
]

# Train the models, generate forecasts, and evaluation
best_model_list = list()
for model, model_name in zip(models_list, models_names_list):
    mase_score, mae_score, rsquare_score = get_model_forecasts(
        model=model, model_name=model_name)
    best_model_list.append(
        {
            'Model name': model_name,
            'MASE': round(mase_score, 2),
            'MAE': round(mae_score, 2),
            'R²': round(rsquare_score, 2)
        }
    )
best_model_dataset = pd.DataFrame(data=best_model_list)
best_model_dataset = best_model_dataset.sort_values(
    by=['MASE', 'MAE'], ascending=True)
print(f'\nBest model dataset:\n{best_model_dataset}')

# Save the best model DataFrame in CSV format
OUTPUT_CSV_2 = 'datasets/best_model_dataset.csv'
best_model_dataset.to_csv(OUTPUT_CSV_2, index=False)


# 3.2 PyCaret
print(f'\n\n3.2 PyCaret')

# Convert the Date column to PeriodIndex
train_dataset = train.to_dataframe()
train_dataset.index = pd.PeriodIndex(train_dataset.index, freq='M')

# Instantiate the model
pycaret_automl = TimeSeriesExperiment(
    target='Consumption (GWh)',
    fh=len(test),
    seasonal_period=PERIOD,
    session_id=SEED
)

# Train the model
pycaret_automl.fit(train_dataset)

# Selection of the best model by cross-validation
results = pycaret_automl.compare_models(round=3, verbose=True)
print(f'\nClassification of models:\n{results.leaderboard}')

# Display information about the best model
best = results.best
print(f'\nBest model:\n{best}')

# Generate forecasts
forecasts_result = pycaret_automl.predict_model(estimator=best)
forecasts_dataset = forecasts_result.predictions
forecasts_dataset = forecasts_dataset.rename(
    columns={'y_pred': 'Consumption (GWh)'})
forecasts_dataset['Date'] = test.time_index.tolist()

# Create the Darts forecasts series from the Pandas DataFrame
forecasts = TimeSeries.from_dataframe(
    df=forecasts_dataset,
    time_col='Date',
    value_cols=['Consumption (GWh)']
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

# Model persistence (save the pipeline)
save_model(best, 'models/monthly consumption/pycaret/model')
