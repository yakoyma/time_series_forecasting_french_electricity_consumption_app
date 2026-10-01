"""
===============================================================================
Time Series Project: Applications for Forecasting the Daily and Monthly
Electricity Consumption in Metropolitan France Excluding Corsica
===============================================================================

This file is organised as follows:
1. Load and explore raw datasets
2. Cleanse the raw daily and raw monthly datasets
   2.1 Cleanse the raw daily dataset
   2.2 Cleanse the raw monthly dataset
3. Statistical tests
   3.1 Statistical tests on the daily dataset
   3.2 Statistical tests on the monthly dataset
"""
# Standard library
import platform

# Other libraries
import pandas as pd
import seaborn as sns
import sweetviz as sv
import ydata_profiling
import statsmodels
import darts


from sweetviz import analyze
from ydata_profiling import ProfileReport
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from darts import TimeSeries
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('Pandas: {}'.format(pd.__version__))
print('Seaborn: {}'.format(sns.__version__))
print('Sweetviz: {}'.format(sv.__version__))
print('YData-profiling: {}'.format(ydata_profiling.__version__))
print('Statsmodels: {}'.format(statsmodels.__version__))
print('Darts: {}'.format(darts.__version__))



# Constants
SEED = 0
MAX_ROWS_DISPLAY = 300
MAX_COLUMNS_DISPLAY = 150

# Set the maximum number of rows to display by Pandas
pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)

# Set the default Seaborn style
sns.set_style('whitegrid')



"""
===============================================================================
1. Load and explore raw datasets
===============================================================================
"""
print(f'\n\n\n1. Load and explore raw datasets')


# Paths of datasets
INPUT_CSV_1 = 'datasets/consommation-nationale-quotidienne-brute.csv'
INPUT_CSV_2 = 'datasets/consommation-nationale-mensuelle-brute.csv'


# Load the raw daily dataset
raw_daily_dataset = load_dataset(file_path=INPUT_CSV_1, encoding='utf-8')

# Display the raw daily dataset information and description
dataset_info_description(dataset=raw_daily_dataset, max_rows=150)

# Generate the raw daily dataset report
raw_daily_dataset_report_sv = analyze(source=raw_daily_dataset)
raw_daily_dataset_report_sv.show_html(
    'datasets/raw_daily_dataset_report_sv.html')
raw_daily_dataset_report_ydp = ProfileReport(
    df=raw_daily_dataset, title='Raw Daily Dataset Report')
raw_daily_dataset_report_ydp.to_file(
    'datasets/raw_daily_dataset_report_ydp.html')


# Load the raw monthly dataset
raw_monthly_dataset = load_dataset(file_path=INPUT_CSV_2, encoding='utf-8')

# Display the raw monthly dataset information and description
dataset_info_description(dataset=raw_monthly_dataset, max_rows=50)

# Generate the raw monthly dataset report
raw_monthly_dataset_report_sv = analyze(source=raw_monthly_dataset)
raw_monthly_dataset_report_sv.show_html(
    'datasets/raw_monthly_dataset_report_sv.html')
raw_monthly_dataset_report_ydp = ProfileReport(
    df=raw_monthly_dataset, title='Raw Monthly Dataset Report')
raw_monthly_dataset_report_ydp.to_file(
    'datasets/raw_monthly_dataset_report_ydp.html')



"""
===============================================================================
2. Cleanse the raw daily and raw monthly datasets
===============================================================================
"""
print(f'\n\n\n2. Cleanse the raw daily and raw monthly datasets')


# 2.1 Cleanse the raw daily dataset
print(f'\n\n2.1 Cleanse the raw daily dataset')

daily_dataset = raw_daily_dataset[
    ['Date', 'Consommation brute électricité (MW) - RTE']]
daily_dataset = daily_dataset.rename(
    columns={'Consommation brute électricité (MW) - RTE': 'Consumption (MW)'})

# Calculate the daily average consumption
daily_dataset['Date'] = pd.to_datetime(daily_dataset['Date'])
daily_dataset = daily_dataset.groupby(['Date']).mean().reset_index(drop=False)
daily_dataset = daily_dataset.sort_values(by=['Date'], ascending=True)

# Management of missing data
if daily_dataset.isna().any().any() == True:
    daily_dataset = daily_dataset.dropna()
    daily_dataset.reset_index(inplace=True, drop=True)

# Display dataset information and description
dataset_info_description(dataset=daily_dataset, max_rows=150)

# Create a Darts series from the Pandas DataFrame
daily_series = TimeSeries.from_dataframe(
    df=daily_dataset,
    time_col='Date',
    value_cols=['Consumption (MW)']
)

# Plot the daily series
plt.figure(figsize=(12, 6))
daily_series.plot(color='tab:blue')
plt.title(label=f'Daily Electricity Consumption (MW)')
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()

# Generate the daily dataset report
daily_dataset_report_sv = analyze(source=daily_dataset)
daily_dataset_report_sv.show_html('datasets/daily_dataset_report_sv.html')
daily_dataset_report_ydp = ProfileReport(
    df=daily_dataset, title='Daily Dataset Report')
daily_dataset_report_ydp.to_file('datasets/daily_dataset_report_ydp.html')

# Save the raw daily dataset in CSV format
OUTPUT_CSV_1 = 'datasets/consommation-nationale-quotidienne.csv'
daily_dataset.to_csv(OUTPUT_CSV_1, index=False)


# 2.2 Cleanse the raw monthly dataset
print(f'\n\n2.2 Cleanse the raw monthly dataset')

monthly_dataset = raw_monthly_dataset[['Mois', 'Consommation brute (GWh)']]
monthly_dataset = monthly_dataset.rename(
    columns={
        'Mois': 'Date',
        'Consommation brute (GWh)': 'Consumption (GWh)'
    }
)
monthly_dataset['Date'] = pd.to_datetime(monthly_dataset['Date'])
monthly_dataset = monthly_dataset.sort_values(by=['Date'], ascending=True)

# Management of missing data
if monthly_dataset.isna().any().any() == True:
    monthly_dataset = monthly_dataset.dropna()
    monthly_dataset.reset_index(inplace=True, drop=True)

# Display dataset information and description
dataset_info_description(dataset=monthly_dataset, max_rows=50)

# Create a Darts series from the Pandas DataFrame
monthly_series = TimeSeries.from_dataframe(
    df=monthly_dataset,
    time_col='Date',
    value_cols=['Consumption (GWh)']
)

# Plot the monthly series
plt.figure(figsize=(12, 6))
monthly_series.plot(color='tab:red')
plt.title(label=f'Monthly Electricity Consumption (GWh)')
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()

# Generate the monthly dataset report
monthly_dataset_report_sv = analyze(source=monthly_dataset)
monthly_dataset_report_sv.show_html('datasets/monthly_dataset_report_sv.html')
monthly_dataset_report_ydp = ProfileReport(
    df=monthly_dataset, title='Monthly Dataset Report')
monthly_dataset_report_ydp.to_file('datasets/monthly_dataset_report_ydp.html')

# Save the raw monthly dataset in CSV format
OUTPUT_CSV_2 = 'datasets/consommation-nationale-mensuelle.csv'
monthly_dataset.to_csv(OUTPUT_CSV_2, index=False)



"""
===============================================================================
3. Statistical tests
===============================================================================
"""
print(f'\n\n\n3. Statistical tests')


# 3.1 Statistical tests on the daily dataset
print(f'\n\n3.1 Statistical tests on the daily dataset')

# Determine whether the daily dataset is stationary
X_daily = daily_dataset['Consumption (MW)'].values
adfuller_test_result = adfuller(x=X_daily, maxlag=365, result_object=True)
print(f'\n\nAugmented Dickey Fuller (ADF) test result: '
      f'{adfuller_test_result}')
kpss_test_result = kpss(x=X_daily, nlags=365, result_object=True)
print(f'\nKwiatkowski-Phillips-Schmidt-Shin (KPSS) test result: '
      f'{kpss_test_result}')

# Plot autocorrelation and partial autocorrelation for the daily dataset
plt.figure(figsize=(12, 6))
plt.plot(
    acf(x=X_daily, nlags=365),
    label='Autocorrelation',
    color='tab:blue'
)
plt.plot(
    pacf(x=X_daily, nlags=365),
    label='Partial Autocorrelation',
    color='tab:red'
 )
plt.title('Autocorrelation and Partial Autocorrelation')
plt.xlabel('Lags')
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()

# Detrend the daily dataset using first-order differentiation
daily_dataset['Consumption (MW)'] = daily_dataset['Consumption (MW)'].diff(
    periods=1)
daily_dataset = daily_dataset.dropna()
daily_dataset.reset_index(inplace=True, drop=True)

# Determine whether the detrended daily dataset is stationary
X_daily = daily_dataset['Consumption (MW)'].values
adfuller_test_result = adfuller(x=X_daily, maxlag=365, result_object=True)
print(f'\n\nAugmented Dickey Fuller (ADF) test result: '
      f'{adfuller_test_result}')
kpss_test_result = kpss(x=X_daily, nlags=365, result_object=True)
print(f'\nKwiatkowski-Phillips-Schmidt-Shin (KPSS) test result: '
      f'{kpss_test_result}')

# Plot autocorrelation and partial autocorrelation for 
# the detrended daily dataset
plt.figure(figsize=(12, 6))
plt.plot(
    acf(x=X_daily, nlags=365),
    label='Autocorrelation',
    color='tab:blue'
)
plt.plot(
    pacf(x=X_daily, nlags=365),
    label='Partial Autocorrelation',
    color='tab:red'
 )
plt.title('Autocorrelation and Partial Autocorrelation')
plt.xlabel('Lags')
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()

# Create a Darts series from the Pandas DataFrame
daily_series = TimeSeries.from_dataframe(
    df=daily_dataset,
    time_col='Date',
    value_cols=['Consumption (MW)']
)

# Plot the daily series
plt.figure(figsize=(12, 6))
daily_series.plot(color='tab:blue')
plt.title(label=f'Daily Electricity Consumption (MW)')
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()


# 3.2 Statistical tests on the monthly dataset
print(f'\n\n3.2 Statistical tests on the monthly dataset')

# Determine whether the monthly dataset is stationary
X_monthly = monthly_dataset['Consumption (GWh)'].values
adfuller_test_result = adfuller(x=X_monthly, maxlag=24, result_object=True)
print(f'\n\nAugmented Dickey Fuller (ADF) test result: '
      f'{adfuller_test_result}')
kpss_test_result = kpss(x=X_monthly, nlags=24, result_object=True)
print(f'\nKwiatkowski-Phillips-Schmidt-Shin (KPSS) test result: '
      f'{kpss_test_result}')

# Plot autocorrelation and partial autocorrelation for the monthly dataset
plt.figure(figsize=(12, 6))
plt.plot(
    acf(x=X_monthly, nlags=24),
    label='Autocorrelation',
    color='tab:blue'
)
plt.plot(
    pacf(x=X_monthly, nlags=24),
    label='Partial Autocorrelation',
    color='tab:red'
 )
plt.title('Autocorrelation and Partial Autocorrelation')
plt.xlabel('Lags')
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()

# Detrend the monthly dataset using first-order differentiation
monthly_dataset['Consumption (GWh)'] = monthly_dataset[
    'Consumption (GWh)'] - monthly_dataset['Consumption (GWh)'].shift(1)
monthly_dataset = monthly_dataset.dropna()
monthly_dataset.reset_index(inplace=True, drop=True)

# Determine whether the detrended monthly dataset is stationary
X_monthly = monthly_dataset['Consumption (GWh)'].values
adfuller_test_result = adfuller(x=X_monthly, maxlag=12, result_object=True)
print(f'\n\nAugmented Dickey Fuller (ADF) test result: '
      f'{adfuller_test_result}')
kpss_test_result = kpss(x=X_monthly, nlags=12, result_object=True)
print(f'\nKwiatkowski-Phillips-Schmidt-Shin (KPSS) test result: '
      f'{kpss_test_result}')

# Plot autocorrelation and partial autocorrelation for 
# the detrended monthly dataset
plt.figure(figsize=(12, 6))
plt.plot(
    acf(x=X_monthly, nlags=12),
    label='Autocorrelation',
    color='tab:blue'
)
plt.plot(
    pacf(x=X_monthly, nlags=12),
    label='Partial Autocorrelation',
    color='tab:red'
 )
plt.title('Autocorrelation and Partial Autocorrelation')
plt.xlabel('Lags')
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()

# Create a Darts series from the Pandas DataFrame
monthly_series = TimeSeries.from_dataframe(
    df=monthly_dataset,
    time_col='Date',
    value_cols=['Consumption (GWh)']
)

# Plot the monthly series
plt.figure(figsize=(12, 6))
monthly_series.plot(color='tab:red')
plt.title(label=f'Monthly Electricity Consumption (GWh)')
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()
