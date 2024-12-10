import numpy as np
import pandas as pd
import pandas_datareader as data
import yfinance as yf
from datetime import datetime
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout='wide')

start = '2010-01-01'
end = datetime.today().strftime('%Y-%m-%d')

st.title('Stock Trend Prediction Web Application')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

if st.button('Analyse'):

    df = yf.download(user_input, start, end)
    st.subheader('Data from 2010 - Present')
    st.write(df.describe())

    st.subheader('Closing Price vs Time Chart')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price', line=dict(color='blue')))
    fig.update_layout(title='Closing Price vs Time', xaxis_title='Time', yaxis_title='Price', template='plotly_dark')
    st.plotly_chart(fig)

    st.subheader('Closing Price vs Time Chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=ma100, mode='lines', name='MA100', line=dict(color='red')))
    fig.update_layout(title='Closing Price with 100MA', xaxis_title='Time', yaxis_title='Price', template='plotly_dark')
    st.plotly_chart(fig)

    st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
    ma200 = df.Close.rolling(200).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=ma100, mode='lines', name='MA100', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=ma200, mode='lines', name='MA200', line=dict(color='green')))
    fig.update_layout(title='Closing Price with 100MA & 200MA', xaxis_title='Time', yaxis_title='Price', template='plotly_dark')
    st.plotly_chart(fig)

    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    model = load_model('keras_model.h5')

    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test, y_test = [], []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)
    scaler = scaler.scale_
    scale_factor = 1 / scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    st.subheader('Predictions vs Original')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index ,y=y_test, mode='lines', name='Original Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=y_predicted.flatten(), mode='lines', name='Predicted Price', line=dict(color='red', dash='dash')))
    fig.update_layout(title='Predictions vs Original', xaxis_title='Time', yaxis_title='Price', template='plotly_dark')
    st.plotly_chart(fig)
