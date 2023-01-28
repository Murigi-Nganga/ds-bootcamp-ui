# import the necessary libraries

# creating dates
from datetime import date, datetime

# ui
import streamlit as st

# preprocessing
import pandas as pd
import numpy as np

# plotting
import matplotlib.pyplot as plt

# using joblib models
import pickle

# import the model to be used
BAC_base_model = pickle.load(open('BAC_base_model.sav', 'rb'))

# import the bank data
df = pd.read_pickle('all_banks.pickle')

# Get a copy where we can reset the index
bank_data = df.copy()
bank_data.columns = ['.'.join(c) for c in bank_data.columns]
bank_data.columns = bank_data.columns.astype('str')

def preprocess(input_data):
    """Function to preprocess the input data"""
    pass
    


# set the page title configurations
st.set_page_config(page_title='Stock Market Analysis App',
                   page_icon='🦈')

st.title('Stock Market Analysis Application')

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
                                     if stock_info_option in i]
                                    ])

        else:
            for option in stock_info_options:
                st.line_chart(bank_data[[i
                                     for i in bank_data.columns.values.tolist()
                                     if option in i]
                                        ])

else:
    st.write('Analysis page')
    
    technique = st.sidebar.selectbox(
        label='Select a technique', 
        options=['Forecasting', 'Prediction']
    )
    
    bank_option_als = st.sidebar.selectbox(
            label='Select Bank', 
            options=banks[:-1]
    )
    
    if technique == 'Forecasting':
        forecast_days = st.sidebar.number_input(
            label='Number fo days to forecast',
            step=1,
            min_value=1
        )
        
    
    else:
        prediction_img, params = st.columns(2)
        # Sidebar with values that will be used for predicting the closing price
        
        prediction_img.image('prediction_img.jpg', caption='Regression prediction')
        
        open_value = st.sidebar.number_input(
            label='Open Value',
            step= 0.01,
            min_value=0.00,
            format="%.2f"
        )
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
        volume_value = st.sidebar.number_input(
            label='Volume Value',
            step= 0.01,
            min_value=0.00,
            format="%.2f"
        )
        
        params_dict = {
            "Open Value" : round(open_value, 2),
            "High Value": round(high_value, 2),
            "Low Value": round(low_value, 2),
            "Volume Value": round(volume_value, 2),
        }

        params.write(params_dict)
        
        features = [open_value, high_value, low_value, volume_value]
        
        result = BAC_base_model.predict([features])
        
        if(st.button("Calculate Close Value")):
            if(result):
                st.write(f"Predicted Close value: **{round(result[0], 2)}**")  #result[0] because predicted value is an array with one value
                st.success("Stock Close Value calculated successfully")
            else:
                st.error("Something went wrong!")
        
        
