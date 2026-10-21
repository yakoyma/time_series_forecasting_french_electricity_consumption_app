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


def get_statistical_tests_result(X, n_lags: int):
    """This function runs the ADF and KPSS tests, displays their results,
    and plots the ACF and PACF curves.

    Args:
        X (pd.DataFrame): the time series to test
        n_lags (int): the number of lags to use in the tests
    """

    adfuller_test_result = adfuller(x=X, maxlag=n_lags, result_object=True)
    print(f'\nAugmented Dickey Fuller (ADF) test result: '
          f'{adfuller_test_result}')
    kpss_test_result = kpss(x=X, nlags=n_lags, result_object=True)
    print(f'\nKwiatkowski-Phillips-Schmidt-Shin (KPSS) test result: '
          f'{kpss_test_result}')

    kpss_pvalue = kpss_test_result.pvalue
    adfuller_pvalue = adfuller_test_result.pvalue
    if kpss_pvalue < 0.05 and adfuller_pvalue >= 0.05:
        print('The series is not stationary.')
    elif kpss_pvalue >= 0.05 and adfuller_pvalue < 0.05:
        print('The series is stationary.')
    elif kpss_pvalue >= 0.05 and adfuller_pvalue >= 0.05:
        print('The series is trend stationary.')
    elif kpss_pvalue < 0.05 and adfuller_pvalue < 0.05:
        print('The series is difference stationary.')

    # Plot autocorrelation and partial autocorrelation
    plt.figure(figsize=(12, 6))
    plt.plot(
        acf(x=X, nlags=n_lags),
        label='Autocorrelation',
        color='tab:blue'
    )
    plt.plot(
        pacf(x=X, nlags=n_lags),
        label='Partial Autocorrelation',
        color='tab:red'
    )
    plt.title('Autocorrelation and Partial Autocorrelation')
    plt.xlabel('Lags')
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()


# Determine whether the daily dataset is stationary
print(f'\n\nDetermine whether the daily dataset is stationary')
get_statistical_tests_result(
    X=daily_dataset['Consumption (MW)'].values, n_lags=365)

# Detrend the daily dataset using first-order differentiation
daily_dataset['Consumption (MW)'] = daily_dataset['Consumption (MW)'].diff(
    periods=1)
daily_dataset = daily_dataset.dropna()
daily_dataset.reset_index(inplace=True, drop=True)

# Determine whether the detrended daily dataset is stationary
print(f'\nDetermine whether the detrended daily dataset is stationary')
get_statistical_tests_result(
    X=daily_dataset['Consumption (MW)'].values, n_lags=365)

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


# Determine whether the monthly dataset is stationary
print(f'\n\nDetermine whether the monthly dataset is stationary')
get_statistical_tests_result(
    X=monthly_dataset['Consumption (GWh)'].values, n_lags=12)

# Detrend the monthly dataset using first-order differentiation
monthly_dataset['Consumption (GWh)'] = monthly_dataset[
    'Consumption (GWh)'] - monthly_dataset['Consumption (GWh)'].shift(1)
monthly_dataset = monthly_dataset.dropna()
monthly_dataset.reset_index(inplace=True, drop=True)

# Determine whether the detrended monthly dataset is stationary
print(f'\n\nDetermine whether the detrended monthly dataset is stationary')
get_statistical_tests_result(
    X=monthly_dataset['Consumption (GWh)'].values, n_lags=12)

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
