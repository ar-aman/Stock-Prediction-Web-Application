import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf #librart for yahoo finance stock data
from datetime import datetime
from keras.models import load_model
import streamlit as st

st.set_page_config(layout='wide')

start = '2010-01-01'
end = datetime.today().strftime('%Y-%m-%d')

st.title('Stock Trend Prediction Web Application')


user_input = st.text_input('Enter Stock Ticker', 'AAPL')

if st.button('Fetch Data'):

    df= yf.download(user_input, start, end)

    # Describing Data
    st.subheader('Data from 2010 - Present')
    st.write(df.describe())

    # Visualizations

    plt.rcParams.update({'axes.facecolor':'#1B1B1C'})

    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Close, 'b', label = 'Closing Price')
    plt.legend(loc='upper left')
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Close, 'b', label = 'Closing Price')
    plt.plot(ma100, 'r', label = 'MA100')
    plt.legend(loc='upper left')
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Close, 'b', label = 'Closing Price')
    plt.plot(ma100, 'r', label = 'MA100')
    plt.plot(ma200, '#D1FF17', label = 'MA100')
    plt.legend()
    st.pyplot(fig)

    #splitting data into training and testing

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

    #importing sklearn to scale down the value between 0 and 1
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)

    #loading model
    model = load_model('keras_model.h5')

    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    x_test,y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)

    #scaling up the value
    scaler = scaler.scale_

    scale_factor = 1/scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    #final graph
    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    st.pyplot(fig2)
