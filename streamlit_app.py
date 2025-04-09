import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import tensorflow as tf
import streamlit as st
import yfinance as yf

start = '2010-01-01'
end = '2022-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
data = yf.download(user_input, start=start, end=end)

#Describing Data

st.subheader('Data from 2010 - 2022')
st.write(data.describe())

#Visualizations

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(data.Close)
st.pyplot(fig)


#Splitting data into Training and Testing

data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#Load my model

model = tf.keras.models.load_model('keras_model.h5')

#Testing part

past_100_days = data_training.tail(100)

final_data = pd.concat([past_100_days, data_testing],ignore_index = True)

input_data = scaler.fit_transform(final_data)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
#Make Predictions

y_predict = model.predict(x_test)
scaler = scaler.scale_


scale_factor = 1/scaler[0]
y_predict = y_predict*scale_factor
y_test = y_test * scale_factor

#Final greaph

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predict, 'r', label ='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
