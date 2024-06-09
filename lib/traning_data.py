import datetime as dt
import yfinance as yf
import numpy as np

class TrainingData:
    """
    Class to generate data using yfinance
    """
    def __init__(self, stock, start_date, end_date, split_percentage, algorithm_settings, log):
        self.symbol_mapping = {
            "Bitcoin": 'BTC-USD',
            "Ethereum": 'ETH-USD',
            "Ripple": 'XRP-USD',
            "Litecoin": 'LTC-USD',
            "Dogecoin": 'DOGE-USD',
            "Gold": 'GC=F'  # Example mapping for Gold
        }

        self.filtered_data = None
        self.data_frame = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.split_percentage = split_percentage / 100
        self.split_index = None
        self.training_by = algorithm_settings.training_by
        self.predicted_by = algorithm_settings.predicted_by
        self.log = log
        self.symbol = self.symbol_mapping[stock]

        self.since_date = start_date
        self.until_date = end_date
    
    def generate(self):
        """
        Function to generate data set
        """
       
        self.data_frame = yf.download(self.symbol, start=self.since_date, end=self.until_date)


    def save_data_frame(self):
        """ Function to print data set and save to a file """
        print(self.data_frame)

        # Save DataFrame to a CSV file
        self.data_frame.to_csv(f'in/chart_{self.symbol}.csv')

    def scale_date(self):
        """
        Function to scale data in range (0,1)
        """
        data = None
        #print(self.data_frame)
        print(type(self.training_by))
        if str(self.training_by) != "date":
            min_date = self.data_frame[self.training_by].values.reshape(-1, 1).min()
            max_date = self.data_frame[self.training_by].values.reshape(-1, 1).max()
            data = self.data_frame[self.training_by].values.reshape(-1, 1)
        else:
            min_date = self.data_frame.index.values.reshape(-1, 1).min()
            max_date = self.data_frame.index.values.reshape(-1, 1).max()
            data = self.data_frame.index.values.reshape(-1, 1)
        window_index = 10
        normalized_dates = (data - min_date) / (max_date - min_date)
        print(self.predicted_by)
        if self.predicted_by != "date":
            min_date = self.data_frame[str(self.predicted_by)].values.reshape(-1, 1).min()
            max_date = self.data_frame[str(self.predicted_by)].values.reshape(-1, 1).max()
            data = self.data_frame[str(self.predicted_by)].values.reshape(-1, 1)
        else:
            min_date = self.data_frame.index.values.reshape(-1, 1).min()
            max_date = self.data_frame.index.values.reshape(-1, 1).max()
            data = self.data_frame.index.values.reshape(-1, 1)

        normalized_price = (data - min_date) / (max_date - min_date)

        #self.split_index_dates = int(0.6 * len(normalized_dates))
        self.split_index = int(self.split_percentage * len(normalized_dates))
        self.x_train, self.y_train = normalized_dates[:self.split_index], normalized_price[:self.split_index]
        self.x_test, self.y_test = normalized_dates[self.split_index:], normalized_price[self.split_index:]
