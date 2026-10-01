"""
===============================================================================
Time Series Project: Application for Forecasting Monthly Electricity
Consumption in Metropolitan France Excluding Corsica using Darts Library
===============================================================================
"""
# Standard library
import platform

# Other libraries
import pandas as pd
import darts
import gradio as gr


from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from darts.models import AutoARIMA
from gradio import Interface, Dropdown, Textbox, LinePlot


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('Pandas: {}'.format(pd.__version__))
print('Darts: {}'.format(darts.__version__))
print('Gradio: {}'.format(gr.__version__))



def get_forecasts(user_year: int, user_month: int):
    """This function loads a pre-trained model and generates monthly forecasts
    of electricity consumption.

    Args:
        user_month (int): the month the forecasts end
        user_year (int): the year the forecasts end

    Returns:
        response (str): the text to be displayed to the user
        dataset (pd.DataFrame): the Pandas DataFrame containing the forecasts
    """

    try:
        
        # Load the train dataset
        INPUT_CSV = 'datasets/monthly consumption/train_dataset.csv'
        train_dataset = pd.read_csv(INPUT_CSV)
        
        # Retrieve dates and calculate the forecast horizon
        start_date = datetime.strptime(
            train_dataset['Date'].iloc[-1], '%Y-%m-%d').date()
        user_date = str(user_month) + '-' + str(user_year)
        user_date = datetime.strptime(user_date, '%m-%Y').date()
        delta_dates = relativedelta(dt1=user_date, dt2=start_date)
        months_number = delta_dates.months + (delta_dates.years * 12)

        # Load the pre-trained model
        model_path = 'models/monthly consumption/darts/AutoARIMA_model.pkl'
        model = AutoARIMA().load(model_path)

        # Generate forecasts
        forecasts = model.predict(n=months_number)
        
        # Convert the Darts forecasts series into Pandas DataFrame
        forecasts_dataset = forecasts.to_dataframe().reset_index()

        # Retrieve a subset from the current date
        forecasts_dataset = forecasts_dataset[
            forecasts_dataset['Date'].dt.date >= date.today().replace(day=1)]
        forecasts_dataset = forecasts_dataset.rename(
            columns={'Consumption (GWh)': 'Consumption'})
        response = (f'Here is the forecasts for monthly electricity '
                    f'consumption in France up to '
                    f'{user_date.strftime("%m-%Y")}.')

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
        Textbox(label='Monthly forecasts result'),
        LinePlot(
            x='Date',
            y='Consumption',
            x_title='Date',
            y_title='Consumption (GWh)'
        )
    ],
    title=('Application for Forecasting Monthly Electricity Consumption in '
           'Metropolitan France Excluding Corsica')
)



if __name__ == '__main__':
    app.launch()
