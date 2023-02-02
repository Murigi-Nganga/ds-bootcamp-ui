# import the necessary libraries

# creating dates
from datetime import date, datetime

# ui
import streamlit as st

# preprocessing
import pandas as pd
import numpy as np

# plotting
import matplotlib
import matplotlib.pyplot as plt

# using models
import pickle

# ARIMA model
from statsmodels.tsa.arima_model import ARIMA

# import the model to be used
BAC_model = pickle.load(open('models/BAC_model.sav', 'rb'))
C_model = pickle.load(open('models/C_model.sav', 'rb'))
GS_model = pickle.load(open('models/GS_model.sav', 'rb'))
JPM_model = pickle.load(open('models/JPM_model.sav', 'rb'))
MS_model = pickle.load(open('models/MS_model.sav', 'rb'))
WFC_model = pickle.load(open('models/WFC_model.sav', 'rb'))

# import the bank data
df = pd.read_pickle('all_banks.pickle')

# Get a copy where we can reset the index
bank_data = df.copy()
bank_data.columns = ['.'.join(c) for c in bank_data.columns]
bank_data.columns = bank_data.columns.astype('str')
    

# set the page title configurations
st.set_page_config(page_title='Stock Market Analysis App',
                   page_icon='🦈')

st.subheader('Stock Market Analysis Application')

st.write('\n')

banks = [('Bank of America', 'BAC'), ('CitiGroup', 'C'),
             ('Goldman Sachs', 'GS'), ('JPMorgan Chase', 'JPM'),
             ('Morgan Stanley', 'MS'), ('Wells Fargo', 'WFC'), 'All']

stock_info_options = ['Open', 'Close', 'High', 'Low', 'Volume', 'All']


page_option = st.sidebar.selectbox(label='Select a page',
                                   options=['Visualizations', 'Analysis'])

st.sidebar.header('Specify the stock input parameters')

st.sidebar.write('---')

if page_option == 'Visualizations':
    
    bank_option_viz = st.sidebar.selectbox(label='Select a bank',
                                       options=banks)

    stock_info_option = st.sidebar.selectbox(label='Stock information option',
                                             options=stock_info_options)

    if bank_option_viz != 'All':

        if stock_info_option == 'All':
            st.line_chart(
                bank_data[[f'{bank_option_viz[1]}.{i}' for i in stock_info_options[:4]]])
            st.line_chart(bank_data[[f'{bank_option_viz[1]}.Volume']])
        
        else:
            st.line_chart(bank_data[[f'{bank_option_viz[1]}.{stock_info_option}']])

    else:
        # check stock option values
        if stock_info_option != 'All':
            print(bank_data.columns.values.tolist())

            st.line_chart(bank_data[[i
                                    for i in bank_data.columns.values.tolist()
                                    if stock_info_option in i
                                    ]])

        else:
            for option in stock_info_options:
                st.line_chart(bank_data[[i
                                    for i in bank_data.columns.values.tolist()
                                    if option in i
                                    ]])

else:
    technique = st.sidebar.selectbox(
        label='Select a technique', 
        options=['Forecasting', 'Prediction']
    )
    
    bank_option_als = st.sidebar.selectbox(
            label='Select Bank', 
            options=banks[:-1]
    )
    
    if technique == 'Forecasting':
        
        # The training percentage to be used to train the model
        training_data_percentage = st.sidebar.number_input(
            label='Training data percentage',
            step=1,
            min_value=1,
            value=90,
            max_value=100
        )
        
        # Number of auto-regressive terms
        p = st.sidebar.number_input(
            label='P Value',
            step=1,
            value=2,
            min_value=0,
            max_value=10
        )
        
        # Number of nonseasonal differences needed for stationarity
        d = st.sidebar.number_input(
            label='D value',
            step=1,
            value=2,
            min_value=0,
            max_value=2
        )
        
        # Number of lagged forecast errors in the prediction equation
        q = st.sidebar.number_input(
            label='Q Value',
            step=1,
            value=1,
            min_value=0,
            max_value=10
        )
        
        # Select the close price data for the chosen bank
        selected_bank = bank_option_als[1]
        
        df_close = df[selected_bank]['Close']
        
        # Changing the data to become stationary
        df_log = np.log(df_close)
        
        # Get the split percentage
        split_percentage = training_data_percentage * .01
        
        train_data, test_data = df_log[3:int(len(df_log)*split_percentage)], \
                                df_log[int(len(df_log)*split_percentage):]
                                
        model = ARIMA(train_data, order=(p, d, q))   
        fitted = model.fit()
        
        steps = len(df_log) - int(len(df_log)*split_percentage)
        fc, se, conf = fitted.forecast(steps, alpha=0.05)
        
        fc_series = pd.Series(fc, index=test_data.index)
        lower_series = pd.Series(conf[:, 0], index=test_data.index)
        upper_series = pd.Series(conf[:, 1], index=test_data.index)
        
        # Find the log inverse of the arrays
        train_data_inv, test_data_inv, fc_series_inv = [np.exp(array) \
                                            for array in [train_data, test_data, fc_series]]
        print(train_data[:5])
        # Create the dataframe of the results
        df = pd.concat([train_data_inv, test_data_inv, fc_series_inv], axis=1)
        
        df.columns = ['Training data', 'Actual Stock Price', 'Predicted Stock Price']
        
        # Plot the results dataframe on a line chart
        st.line_chart(df, use_container_width=True)
        
        st.sidebar.text('\n\n')
        
    else:
        prediction_img, params = st.columns(2)
        # Sidebar with values that will be used for predicting the closing price
        
        prediction_img.image('prediction_img.jpg', caption='Regression prediction')
        
        high_value = st.sidebar.number_input(
            label='High Value',
            step= 0.01,
            min_value=0.00,
            format="%.2f"
        )
        low_value = st.sidebar.number_input(
            label='Low Value',
            step= 0.01,
            min_value=0.00,
            format="%.2f"
        )
        open_value = st.sidebar.number_input(
            label='Open Value',
            step= 0.01,
            min_value=0.00,
            format="%.2f"
        )
        
        
        st.sidebar.text('\n\n')
        
        params_dict = {
            "High Value": round(high_value, 2),
            "Low Value": round(low_value, 2),
            "Open Value" : round(open_value, 2),
        }
             
        params.write(params_dict)
        
        features = [high_value, low_value, open_value]
        
        selected_bank = bank_option_als[1]
        
        if selected_bank == 'BAC':
            result = BAC_model.predict([features])
        elif selected_bank == 'C':
            result = C_model.predict([features])
        elif selected_bank == 'GS':
            result = GS_model.predict([features])
        elif selected_bank == 'JPM':
            result = JPM_model.predict([features])
        elif selected_bank == 'MS':
            result = GS_model.predict([features])
        elif selected_bank == 'WFC':
            result = WFC_model.predict([features])
        
        if low_value > high_value:
            st.error("Error: Low value must be less that high value")
        else:
            if(st.button("Calculate Close Value")):
                if(result):
                    st.write(f"Predicted Close value: **{round(result[0], 2)}**")  #result[0] because predicted value is an array with one value
                    st.success("Stock Close Value calculated successfully")
                else:
                    st.error("Something went wrong!")
        
        
