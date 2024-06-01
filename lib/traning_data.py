"""
    Class represent generation training data
"""
import datetime as dt
import yfinance as yf
import numpy as np

class TrainingData:
    """
       Class to generate date using yfinance
    """
    def __init__(self):
        self.filtered_data = None
        self.data_frame = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.split_index = None

        self.symbol = 'GC=F'

        self.since_date = dt.datetime(2003, 1, 1)
        self.until_date = dt.datetime(2024, 2, 25)
    
    def generate(self):
        """
        Function to generate data set
        """

        # Fetch historical data using yfinance
        self.data_frame = yf.download(self.symbol, start=self.since_date, end=self.until_date)
        #self.data_frame = self.data_frame.resample('h').ffill().bfill()

    def print_data_frame(self):
        """ Function to print data set and save to a file """
        print(self.data_frame)

        # Save DataFrame to a CSV file
        self.data_frame.to_csv('in/chart_btc.csv')

    def scale_date( self ):
        """
        Function to scale data in range (0,1)
        """
        print(self.data_frame)
        min_date = self.data_frame['Open'].values.reshape(-1, 1).min()
        max_date = self.data_frame['Open'].values.reshape(-1, 1).max()
        min_date_index = self.data_frame.index.values.reshape(-1, 1).min()
        max_date_index = self.data_frame.index.values.reshape(-1, 1).max()
        data = self.data_frame['Open'].values.reshape(-1, 1)
        window_index = 10
        normalized_dates = (data - min_date) / (max_date - min_date)


        min_date = self.data_frame['High'].values.reshape(-1, 1).min()
        max_date = self.data_frame['High'].values.reshape(-1, 1).max()

        normalized_price = ( self.data_frame['High'].values.reshape(-1, 1) - min_date) / (max_date - min_date)

        self.split_index_dates = int(0.6 * len(normalized_dates))
        self.split_index = int(0.6 * len(normalized_dates))
        self.x_train, self.y_train = normalized_dates[:self.split_index], normalized_price[:self.split_index_dates]
        self.x_test, self.y_test = normalized_dates[self.split_index:], normalized_price[self.split_index_dates:]