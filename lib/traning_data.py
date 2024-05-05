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

        self.since_date = dt.datetime(2002, 1, 1)
        self.until_date = dt.datetime(2024, 2, 25)

    def generate(self):
        """
        Function to generate data set
        """

        # Fetch historical data using yfinance
        self.data_frame = yf.download(self.symbol, start=self.since_date, end=self.until_date)

    def print_data_frame(self):
        """ Function to print data set and save to a file """
        print(self.data_frame)

        # Save DataFrame to a CSV file
        self.data_frame.to_csv('in/chart_btc.csv')

    def scale_date( self ):
        """
        Function to scale data in range (0,1)
        """
        data = self.data_frame['Close'].values.reshape(-1, 1) - np.min(self.data_frame)
        normalized_data = (data / (np.max(data) - np.min(data)))  # Normalize dat
        x = normalized_data[:-1]
        y = normalized_data[1:]
        # Split the data into training and testing sets (80-20 split)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

        self.split_index = int(0.8 * len(x))
        self.x_train, self.y_train = x[:self.split_index], y[:self.split_index]
        self.x_test, self.y_test = x[self.split_index:], y[self.split_index:]
