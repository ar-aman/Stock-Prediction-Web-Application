import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import pandas_datareader as data
#import yfinance as yf #library for yahoo finance stock data
from datetime import datetime
start = '2010-01-01'
end = datetime.today().strftime('%Y-%m-%d')

df= yf.download('AAPL', start, end)
df.head()
