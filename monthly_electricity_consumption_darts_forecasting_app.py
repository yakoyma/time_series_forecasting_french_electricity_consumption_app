"""
===============================================================================
Time Series Project: Application for Forecasting the Monthly Electricity
Consumption in Metropolitan France Excluding Corsica using Darts Library
===============================================================================
"""
# Standard library
import platform

# Other libraries
import pandas as pd
import darts
import joblib
import gradio as gr


from datetime import datetime, date
from dateutil.relativedelta import relativedelta
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
from joblib import load
from gradio import Interface, Dropdown, Textbox, LinePlot


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('Pandas: {}'.format(pd.__version__))
print('Darts: {}'.format(darts.__version__))
print('Joblib: {}'.format(joblib.__version__))
print('Gradio: {}'.format(gr.__version__))



def get_forecasts(user_year: int, user_month: int):
    """This function loads a pre-trained model and generates the monthly
    electricity consumption forecasts for Gradio application.

    Args:
        user_month (int): the month the forecasts end
        user_year (int): the year the forecasts end

    Returns:
        response (str): the text to be displayed to the user
        dataset (pd.DataFrame): the Pandas DataFrame containing the forecasts
    """

    try:
        
        # Load the train dataset
        INPUT_CSV_1 = 'datasets/monthly consumption/train_dataset.csv'
        train_dataset = pd.read_csv(INPUT_CSV_1)
        
        # Retrieve dates and calculate the forecast horizon
        start_date = datetime.strptime(
            train_dataset['Date'].iloc[-1], '%Y-%m-%d').date()
        user_date = str(user_month) + '-' + str(user_year)
        user_date = datetime.strptime(user_date, '%m-%Y').date()
        delta_dates = relativedelta(dt1=user_date, dt2=start_date)
        months_number = delta_dates.months + (delta_dates.years * 12)

        # Load the pre-trained model
        INPUT_CSV_2 = 'datasets/best_model_dataset.csv'
        best_model_dataset = pd.read_csv(INPUT_CSV_2)
        model_name = best_model_dataset['Model name'].iloc[0]
        model_path = f'models/monthly consumption/darts/{model_name}_model.pkl'
        model = globals()[model_name].load(model_path)

        # Generate forecasts
        forecasts_scaled = model.predict(n=months_number)

        # Reverse normalisation
        scaler_path = 'models/scaler/monthly consumption/darts/scaler.joblib'
        scaler = load(filename=scaler_path)
        forecasts = scaler.inverse_transform(series=forecasts_scaled)
        
        # Convert the Darts forecasts series into Pandas DataFrame
        forecasts_dataset = forecasts.to_dataframe().reset_index()

        # Retrieve a subset from the current date
        forecasts_dataset = forecasts_dataset[
            forecasts_dataset['Date'].dt.date >= date.today().replace(day=1)]
        forecasts_dataset = forecasts_dataset.rename(
            columns={'Consumption (GWh)': 'Consumption'})
        response = (f'Here are the forecasts for the monthly electricity '
                    f'consumption up to {user_date.strftime("%m-%Y")}.')

    except Exception as error:
        forecasts_dataset = pd.DataFrame()
        response = f'The following error occurred: {error}'
    return response, forecasts_dataset



# Instantiate the app
app = Interface(
    fn=get_forecasts,
    inputs=[
        Dropdown(
            choices=list(range(date.today().year + 1, date.today().year + 3)),
            label='Please select the year in which the forecasts end',
            type='value'
        ),
        Dropdown(
            choices=list(range(1, 13)),
            label='Please select the month in which the forecasts end',
            type='value'
        )
    ],
    outputs=[
        Textbox(label='Forecasts result'),
        LinePlot(
            x='Date',
            y='Consumption',
            title='Monthly electricity consumption forecasts',
            x_title='Date',
            y_title='Consumption (GWh)'
        )
    ],
    title=('Application for Forecasting the Monthly Electricity Consumption '
           'in Metropolitan France Excluding Corsica')
)



if __name__ == '__main__':
    app.launch()
