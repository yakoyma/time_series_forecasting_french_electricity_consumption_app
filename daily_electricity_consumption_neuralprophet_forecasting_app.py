"""
===============================================================================
Time Series Project: Application for Forecasting Daily Electricity Consumption
in Metropolitan France Excluding Corsica using NeuralProphet Library
===============================================================================
"""
# Standard library
import platform

# Other libraries
import pandas as pd
import neuralprophet
import gradio as gr


from datetime import datetime, timedelta, date
from neuralprophet import load
from gradio import Interface, Textbox, LinePlot


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('Pandas: {}'.format(pd.__version__))
print('NeuralProphet: {}'.format(neuralprophet.__version__))
print('Gradio: {}'.format(gr.__version__))



def get_forecasts(user_date: str):
    """This function loads a pre-trained model and generates daily forecasts of
    electricity consumption for Gradio application.

    Args:
        user_date (str): the end date for the forecasts

    Returns:
        response (str): the text to be displayed to the user
        dataset (pd.DataFrame): the Pandas DataFrame containing the forecasts
    """

    try:
        
        # Cleanse the text
        user_date = user_date.strip()
            
        # Load the train dataset
        INPUT_CSV = ('datasets/daily consumption/deepl learning/'
                     'train_dataset.csv')
        train_dataset = pd.read_csv(INPUT_CSV)
        
        # Retrieve dates and calculate the forecast horizon
        start_date = datetime.strptime(
            train_dataset['Date'].max(), '%Y-%m-%d').date() + timedelta(days=1)
        user_date = datetime.strptime(user_date, '%d-%m-%Y').date()
        forecasts_start_date = date.today() + timedelta(days=7)
        days_number = abs(user_date - start_date).days + 1

        # Check whether the date entered by the user is correct
        if ((user_date - forecasts_start_date).days >= 0 and
            user_date.year <= date.today().year + 2):

            # Load the pre-trained model
            model_path = 'models/daily consumption/neuralprophet/model.np'
            model = load(path=model_path)

            # Generate forecasts
            X_train = train_dataset.rename(columns={
                'Date': 'ds', 'Consumption (MW)': 'y'})
            X_test = model.make_future_dataframe(
                df=X_train, periods=days_number)
            forecasts = model.predict(df=X_test)
            forecasts_dataset = forecasts.reset_index()
            forecasts_dataset = forecasts_dataset.rename(columns={
                'ds': 'Date', 'yhat1': 'Consumption'})

            # Retrieve a subset from the current date
            forecasts_dataset = forecasts_dataset[
                forecasts_dataset['Date'].dt.date >= date.today()]
            response = (f'Here is the forecasts for daily electricity '
                        f'consumption in France up to '
                        f'{user_date.strftime("%d-%m-%Y")}.')
        else:
            forecasts_dataset = pd.DataFrame()
            response = (f'The date is incorrect. It must be greater than or '
                        f'equal to {forecasts_start_date.strftime("%d-%m-%Y")} '
                        f'and less than or equal to '
                        f'31-12-{date.today().year + 2}.')

    except Exception as error:
        forecasts_dataset = pd.DataFrame()
        response = f'The following error occurred: {error}'
    return response, forecasts_dataset



# Instantiate the app
forecasts_start_date = date.today() + timedelta(days=7)
forecasts_start_date = forecasts_start_date.strftime('%d-%m-%Y')
app = Interface(
    fn=get_forecasts,
    inputs=[Textbox(
        label = f'Please enter the end date for the forecasts from '
                f'{forecasts_start_date} to 31-12-{date.today().year + 2} in '
                f'the format DD-MM-YYYY'
        )
    ],
    outputs=[
        Textbox(label='Daily forecasts result'),
        LinePlot(
            x='Date',
            y='Consumption',
            x_title='Date',
            y_title='Consumption (MW)'
        )
    ],
    title=('Application for Forecasting Daily Electricity Consumption in '
           'Metropolitan France Excluding Corsica')
)



if __name__ == '__main__':
    app.launch()
