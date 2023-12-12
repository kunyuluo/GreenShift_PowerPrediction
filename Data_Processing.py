import pandas as pd
import numpy as np
from Helper import format_date, transfer_time_zone
from Helper import select_by_date, select_by_time
from Helper import plot_variable, plot_variable_no_time


class GetData:
    def __init__(self,
                 file_path: str,
                 column_name: str = 'cp_power',
                 start_month: int = None,
                 start_day: int = None,
                 end_month: int = None,
                 end_day: int = None,
                 hour_range: tuple = None,
                 group_freq: int = None,
                 test_size: float = 0.2,
                 n_input: int = 5,
                 n_output: int = 5,
                 timestep: int = 5):

        self.file_path = file_path
        self.column_name = column_name
        self.start_month = start_month
        self.start_day = start_day
        self.end_month = end_month
        self.end_day = end_day
        self.hour_range = hour_range
        self.group_freq = group_freq
        self.n_input = n_input
        self.n_output = n_output
        self.timestep = timestep
        self.test_size = test_size
        self.df = self.extract_historic_data()
        self.train, self.test = self.etl()
        self.X_train, self.y_train = self.to_supervised(self.train)
        self.X_test, self.y_test = self.to_supervised(self.test)

    def extract_historic_data(self):
        # Load data file:
        # *******************************************************************************
        data = pd.read_csv(self.file_path, low_memory=False)

        # Time zone transfer from UTC to local ('US/Eastern)
        # *******************************************************************************
        date_local = transfer_time_zone(data)
        # date_local = format_date(data)

        # Get data from specific column
        # *******************************************************************************
        target_data = pd.concat([date_local, data[self.column_name]], axis=1)

        # Get data for specified period
        # *******************************************************************************
        if (self.start_month is not None and
                self.start_day is not None and
                self.end_month is not None and
                self.end_day is not None):
            target_period = select_by_date(
                target_data, self.start_month, self.start_day, self.end_month, self.end_day)
        else:
            target_period = target_data

        if self.hour_range is not None:
            target_period = select_by_time(target_period, self.hour_range[0], self.hour_range[1])
        else:
            target_period = target_period

        if self.group_freq is not None:
            target_period = target_period.groupby(pd.Grouper(freq=f'{self.group_freq}min')).mean()

        target_period = target_period.dropna()
        # print(target_period)

        return target_period

    def split_data(self):
        """
        Split data into train and test sets.
        """
        data = self.extract_historic_data()

        if len(data) != 0:
            train_idx = round(len(data) * (1 - self.test_size))
            train = data[:train_idx]
            test = data[train_idx:]
            train = np.array(train[self.column_name])
            test = np.array(test[self.column_name])
            return train[:, np.newaxis], test[:, np.newaxis]
        else:
            raise Exception('Data set is empty, cannot split.')

    def etl(self) -> tuple[np.array, np.array]:
        """
        Runs complete ETL
        """
        train, test = self.split_data()
        return self.transform(train, test)

    def transform(self, train: np.array, test: np.array):

        train_remainder = train.shape[0] % self.timestep
        test_remainder = test.shape[0] % self.timestep

        if train_remainder != 0 and test_remainder != 0:
            train = train[0: train.shape[0] - train_remainder]
            test = test[0: test.shape[0] - test_remainder]
            # train = train[train_remainder:]
            # test = test[test_remainder:]
        elif train_remainder != 0:
            train = train[0: train.shape[0] - train_remainder]
        elif test_remainder != 0:
            test = test[0: test.shape[0] - test_remainder]

        return self.window_and_reshape(train), self.window_and_reshape(test)

    def window_and_reshape(self, data) -> np.array:
        """
        Reformats data into shape our model needs,
        namely, [# samples, timestep, # feautures]
        samples
        """
        num_features = 1
        samples = int(data.shape[0] / self.timestep)
        result = np.array(np.array_split(data, samples))
        return result.reshape((samples, self.timestep, num_features))

    def to_supervised(self, train) -> tuple:
        """
        Converts our time series prediction problem to a
        supervised learning problem.
        """
        # flatted the data
        data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
        X, y = [], []
        in_start = 0

        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + self.n_input
            out_end = in_end + self.n_output

            # ensure we have enough data for this instance
            if out_end <= len(data):
                x_input = data[in_start:in_end, 0]
                x_input = x_input.reshape((len(x_input), 1))
                X.append(x_input)
                y.append(data[in_end:out_end, 0])

                # move along one time step
                in_start += 1

        return np.array(X), np.array(y)


data = GetData(file_path='gs_real_time_data.csv',
               column_name='cp_power',
               start_month=9,
               start_day=1,
               end_month=10,
               end_day=19,
               hour_range=(8, 20),
               # group_freq=5,
               test_size=0.2,
               n_input=15,
               n_output=10)


print(data.X_train)
print(data.X_train.shape)
print(data.y_train)
print(data.y_train.shape)

