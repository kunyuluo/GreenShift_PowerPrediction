import numpy as np
import pandas as pd
import pytz
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler


def construct_date_columns(df: pd.DataFrame, date_column: str = 'data_time'):
    df['Year'] = df[date_column].dt.year
    df['Month'] = df[date_column].dt.month
    df['Day'] = df[date_column].dt.day
    df['Hour'] = df[date_column].dt.hour
    df['Minute'] = df[date_column].dt.minute

    # power_data.drop(['data_time'], axis=1, inplace=True)
    # column_names = power_data.columns.tolist()
    column_names = ['data_time', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'cp_power']
    df = df[column_names]

    return df


class DefaultValueFiller:
    def __init__(self, df: pd.DataFrame, feature_names=None):

        feature_names = [] if feature_names is None else feature_names

        self.df = df
        self.feature_names = feature_names
        self.datetime_column = 'data_time'
        self.feature_data = self.get_feature_data()
        self.new_dataset = self.fill_missing_value()

    def transfer_time_zone(self, column: str = 'data_time'):
        self.df['data_time'] = pd.to_datetime(self.df[column])
        df_utc = self.df['data_time'].dt.tz_localize('UTC')
        df_local = df_utc.dt.tz_convert(pytz.timezone('US/Eastern')).dt.tz_localize(None).dt.floor('min')

        return df_local

    def get_feature_data(self):
        # Time zone transfer from UTC to local ('US/Eastern)
        # *******************************************************************************
        date_local = self.transfer_time_zone()

        # Get data from specific column
        # *******************************************************************************
        feature_data = pd.concat([date_local, self.df[self.feature_names]], axis=1)
        feature_data['weekday'] = feature_data['data_time'].dt.weekday

        return feature_data

    def fill_missing_value(self):

        # Construct new dataframe without missing value for each timestep
        # *******************************************************************************
        datetimes = []
        dt_min = self.feature_data['data_time'].min()
        dt_max = self.feature_data['data_time'].max()
        year_range = range(dt_min.year, dt_max.year + 1)
        month_range = range(dt_min.month, dt_max.month + 1)
        start_day, end_day = dt_min.day, dt_max.day
        num_days_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        for year in year_range:
            for i, month in enumerate(month_range):
                if i == 0:
                    for day in range(start_day, num_days_month[month - 1] + 1):
                        for hour in range(24):
                            for minute in range(60):
                                datetimes.append(dt.datetime(year=year, month=month, day=day, hour=hour, minute=minute))
                elif i == len(month_range) - 1:
                    for day in range(1, end_day + 1):
                        for hour in range(24):
                            for minute in range(60):
                                datetimes.append(dt.datetime(year=year, month=month, day=day, hour=hour, minute=minute))
                else:
                    for day in range(1, num_days_month[i] + 1):
                        for hour in range(24):
                            for minute in range(60):
                                datetimes.append(dt.datetime(year=year, month=month, day=day, hour=hour, minute=minute))

        new_df = pd.DataFrame(pd.to_datetime(datetimes), columns=['data_time'])
        new_df['weekday'] = new_df['data_time'].dt.weekday

        for feature in self.feature_names:
            default = self.calc_default_value(feature)
            filled_data = []
            for date in datetimes:
                value = self.feature_data[(self.feature_data['data_time'] == date)][feature].values

                if len(value) == 0:
                    weekday = date.weekday()
                    if weekday in [0, 1, 2, 3, 4]:
                        value = default[0][date.hour][date.minute]
                    elif weekday in [5, 6]:
                        value = default[1][date.hour][date.minute]

                    filled_data.append(value)
                else:
                    filled_data.append(value[0])

            # Fill the strange zero value:
            self.fill_strange_value(filled_data)

            new_df[feature] = filled_data
            # new_df.drop(['weekday'], axis=1, inplace=True)

        return new_df

    def fill_strange_value(self, data):
        """
        Replace zero value in the input list with its previous value.
        """
        for i in range(len(data)):
            if data[i] == 0:
                if i == 0:
                    data[i] = data[i + 1]
                else:
                    data[i] = data[i - 1]
            elif data[i] != 0 and data[i-1] != 0 and 1 - (data[i] / data[i-1]) > 0.8:
                data[i] = data[i - 1] * 0.5
            else:
                pass

    def calc_default_value(self, column_name):
        """
            Calculate average value of every minute in a day by weekday (from Monday to Sunday).
            Use the calculated value to fill empty/missing value in the dataset.
        """

        hours = range(24)
        minutes = range(60)
        default_values = {}

        weekdays = {}
        weekends = {}
        for hour in hours:
            hours_wday = []
            hours_wend = []
            for minute in minutes:
                value_wday = self.feature_data[((self.feature_data['weekday'] == 0) |
                                                (self.feature_data['weekday'] == 1) |
                                                (self.feature_data['weekday'] == 2) |
                                                (self.feature_data['weekday'] == 3) |
                                                (self.feature_data['weekday'] == 4)) &
                                               (self.feature_data['data_time'].dt.hour == hour) &
                                               (self.feature_data['data_time'].dt.minute == minute)][column_name].mean()

                value_wend = self.feature_data[((self.feature_data['weekday'] == 5) |
                                                (self.feature_data['weekday'] == 6)) &
                                               (self.feature_data['data_time'].dt.hour == hour) &
                                               (self.feature_data['data_time'].dt.minute == minute)][column_name].mean()
                hours_wday.append(value_wday)
                hours_wend.append(value_wend)
            weekdays[hour] = hours_wday
            weekends[hour] = hours_wend

        default_values[0] = weekdays
        default_values[1] = weekends

        return default_values


class GSDataProcessor:
    def __init__(self,
                 file_path: str,
                 df: pd.DataFrame = None,
                 feature_names=None,
                 start_month: int = None,
                 start_day: int = None,
                 end_month: int = None,
                 end_day: int = None,
                 hour_range: tuple = None,
                 group_freq: int = None,
                 test_size: float = 0.2,
                 n_input: int = 5,
                 n_output: int = 5,
                 time_zone_transfer: bool = False,
                 date_column: str = 'data_time'):

        feature_names = [] if feature_names is None else feature_names

        data = df if df is not None else pd.read_csv(file_path, low_memory=False)

        num_features = len(feature_names) if len(feature_names) > 0 else data.shape[1] - 1

        self.df = data
        self.feature_names = feature_names
        self.start_month = start_month
        self.start_day = start_day
        self.end_month = end_month
        self.end_day = end_day
        self.hour_range = hour_range
        self.group_freq = group_freq
        self.test_size = test_size
        self.n_input = n_input
        self.n_output = n_output
        self.time_zone_transfer = time_zone_transfer
        self.date_column = date_column
        self.num_features = num_features
        self.train, self.test = self.get_train_test()
        self.X_train, self.y_train = self.to_supervised(self.train)
        self.X_test, self.y_test = self.to_supervised(self.test)

    def format_date(self):
        self.df['data_time'] = pd.to_datetime(self.df[self.date_column])
        df_local = self.df['data_time'].dt.tz_localize(None).dt.floor('min')

        return df_local

    def transfer_time_zone(self):
        self.df['data_time'] = pd.to_datetime(self.df[self.date_column])
        df_utc = self.df['data_time'].dt.tz_localize('UTC')
        df_local = df_utc.dt.tz_convert(pytz.timezone('US/Eastern')).dt.tz_localize(None).dt.floor('min')
        # self.df['data_time'] = df_local
        # df_utc = df_utc.dt.tz_localize(None)

        return df_local

    def select_by_date(
            self,
            df: pd.DataFrame,
            start_year: int = dt.date.today().year,
            end_year: int = dt.date.today().year,
            start_month: int = None,
            start_day: int = None,
            end_month: int = None,
            end_day: int = None):

        # year = dt.date.today().year
        # date_start = dt.date(year, start_month, start_day)
        # date_end = dt.date(year, end_month, end_day)
        # days = (date_end - date_start).days + 1
        if (start_month is not None and
                end_month is not None and
                start_day is not None and
                end_day is not None):
            df_selected = df[
                (df[self.date_column].dt.year >= start_year) &
                (df[self.date_column].dt.year <= end_year) &
                (df[self.date_column].dt.month >= start_month) &
                (df[self.date_column].dt.month <= end_month) &
                (df[self.date_column].dt.day >= start_day) &
                (df[self.date_column].dt.day <= end_day)]

            # df_selected.set_index('data_time', inplace=True)

            return df_selected

    def select_by_time(self, df: pd.DataFrame, start_hour: int = 8, end_hour: int = 22):
        df_selected = df[
            (df[self.date_column].dt.hour >= start_hour) &
            (df[self.date_column].dt.hour < end_hour)]

        return df_selected

    def get_period_data(self):
        # Time zone transfer from UTC to local ('US/Eastern)
        # *******************************************************************************
        if self.time_zone_transfer:
            date_local = self.transfer_time_zone()
        else:
            date_local = self.format_date()

        # Get data from specific column
        # *******************************************************************************
        if len(self.feature_names) != 0:
            target_data = pd.concat([date_local, self.df[self.feature_names]], axis=1)
        else:
            feature_df = self.df.drop([self.date_column], axis=1)
            target_data = pd.concat([date_local, feature_df], axis=1)

        # Get data for specified period
        # *******************************************************************************
        if (self.start_month is not None and
                self.start_day is not None and
                self.end_month is not None and
                self.end_day is not None):
            target_period = self.select_by_date(
                target_data, start_month=self.start_month, start_day=self.start_day,
                end_month=self.end_month, end_day=self.end_day)
        else:
            target_period = target_data

        if self.hour_range is not None:
            target_period = self.select_by_time(target_period, self.hour_range[0], self.hour_range[1])
        else:
            target_period = target_period

        target_period.set_index(self.date_column, inplace=True)

        if self.group_freq is not None:
            target_period = target_period.groupby(pd.Grouper(freq=f'{self.group_freq}min')).mean()

        target_period = target_period.dropna()
        # target_period = target_period.reset_index()
        # print(target_period)

        return target_period

    def get_train_test(self) -> tuple[np.array, np.array]:
        """
        Runs complete ETL
        """
        train, test = self.split_data()
        return self.transform(train, test)

    def split_data(self):
        """
        Split data into train and test sets.
        """
        data = self.get_period_data()

        if len(data) != 0:
            train_idx = round(len(data) * (1 - self.test_size))
            train = data[:train_idx]
            test = data[train_idx:]
            # train = np.array(np.split(train, train.shape[0] / self.timestep))
            # test = np.array(np.split(test, test.shape[0] / self.timestep))
            return train.values, test.values
        else:
            raise Exception('Data set is empty, cannot split.')

    def transform(self, train: np.array, test: np.array):

        train_remainder = train.shape[0] % self.n_input
        test_remainder = test.shape[0] % self.n_input

        if train_remainder != 0 and test_remainder != 0:
            # train = train[0: train.shape[0] - train_remainder]
            # test = test[0: test.shape[0] - test_remainder]
            train = train[train_remainder:]
            test = test[test_remainder:]
        elif train_remainder != 0:
            train = train[train_remainder:]
        elif test_remainder != 0:
            test = test[test_remainder:]

        return self.window_and_reshape(train), self.window_and_reshape(test)
        # return train, test

    def window_and_reshape(self, data) -> np.array:
        """
        Reformats data into shape our model needs,
        namely, [# samples, timestep, # feautures]
        samples
        """
        samples = int(data.shape[0] / self.n_input)
        result = np.array(np.array_split(data, samples))
        return result.reshape((samples, self.n_input, self.num_features))

    def to_supervised(self, data) -> tuple:
        """
        Converts our time series prediction problem to a
        supervised learning problem.
        Input has to be reshaped to 3D [samples, timesteps, features]
        """
        # flatted the data
        data_flattened = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        X, y = [], []
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + self.n_input
            out_end = in_end + self.n_output
            # ensure we have enough data for this instance
            if out_end <= len(data_flattened):
                x_input = data_flattened[in_start:in_end, :]
                # x_input = x_input.reshape((x_input.shape[0], x_input.shape[1], 1))
                X.append(x_input)
                y.append(data_flattened[in_end:out_end, 0])
                # move along one time step
                in_start += 1
        return np.array(X), np.array(y)

    @staticmethod
    def check_data_distribution(df: pd.DataFrame, column_name: str = 'cp_power', log_transform: bool = False):
        """
        Check the distribution of the data.
        """
        data = df[column_name]

        if log_transform:
            data = np.log(data)

        plt.hist(data, bins=20)
        plt.title('Distribution of \'{}\''.format(column_name))
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.show()

    @staticmethod
    def check_linearity(
            df: pd.DataFrame,
            column_name_1: str,
            column_name_2: str,
            switch_table: bool = False,
            log_transform: bool = False):
        """
        Check the linearity of the data.
        """
        data_1 = df[column_name_1]
        data_2 = df[column_name_2]

        if log_transform:
            data_1 = np.log(data_1)
            # data_2 = np.log(data_2)

        x_label = column_name_1
        y_label = column_name_2

        if switch_table:
            data_1, data_2 = data_2, data_1
            x_label, y_label = y_label, x_label

        plt.scatter(data_1, data_2, s=2)
        plt.title('Linearity between \'{}\' and \'{}\''.format(column_name_1, column_name_2))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    @staticmethod
    def check_autocorrelation(df: pd.DataFrame, column_name: str = 'cp_power'):
        """
        Check the autocorrelation of the data.
        """
        data = df[column_name]

        pd.plotting.lag_plot(data)
        plt.show()

    @staticmethod
    def plot_variable(df: pd.DataFrame, var_name: str = 'cp_power', is_daily: bool = True):
        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot()

        if is_daily:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        # ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax.plot(df.index, df[var_name], color='black')
        ax.set_ylim(0, )

        plt.show()

    @staticmethod
    def plot_variable_no_time(df: pd.DataFrame, var_name: str = 'cp_power'):
        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot()

        x = range(len(df[var_name]))
        ax.plot(x, df[var_name], color='black')
        ax.set_ylim(0, )

        plt.show()


class PredictAndForecast:
    """
    model: tf.keras.Model
    train: np.array
    test: np.array
    Takes a trained model, train, and test datasets and returns predictions
    of len(test) with same shape.
    """

    def __init__(self, model, train, test, n_input=5, n_output=5) -> None:
        train = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
        test = test.reshape((test.shape[0] * test.shape[1], test.shape[2]))

        self.model = model
        self.train = train
        self.test = test
        self.n_input = n_input
        self.n_output = n_output
        # self.updated_test = self.updated_test()
        # self.predictions = self.get_predictions()

    def forcast(self, x_input) -> np.array:
        """
        Given last weeks actual data, forecasts next weeks' prices.
        """
        # Flatten data
        # data = np.array(history)
        # data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))

        # retrieve last observations for input data
        # x_input = data[-self.n_input:, :]
        # x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))

        # forecast the next week
        yhat = self.model.predict(x_input, verbose=0)

        # we only want the vector forecast
        yhat = yhat[0]
        return yhat

    def get_predictions(self) -> np.array:
        """
        compiles models predictions week by week over entire test set.
        """
        # history is a list of flattened test data + last observation from train data
        test_remainder = self.test.shape[0] % self.n_output
        if test_remainder != 0:
            test = self.test[:-test_remainder]
        else:
            test = self.test

        history = [x for x in self.train[-self.n_input:, :]]
        history.extend(test)

        step = round(len(history) / self.n_output)
        # history = []

        # walk-forward validation
        predictions = []
        window_start = 0
        for i in range(step):

            if window_start <= len(history) - self.n_input - self.n_output:
                # print('pred no {}, window_start {}'.format(i+1, window_start))
                x_input = np.array(history[window_start:window_start + self.n_input])
                x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
                yhat_sequence = self.forcast(x_input)
                # print('pred no {}'.format(i))
                # store the predictions
                predictions.append(yhat_sequence)

            window_start += self.n_output
            # get real observation and add to history for predicting the next week
            # history.append(self.test[i, :])

        return np.array(predictions)

    def updated_test(self):
        test_remainder = self.test.shape[0] % self.n_output
        if test_remainder != 0:
            test = self.test[:-test_remainder]
            return test
        else:
            return self.test

    def get_sample_prediction(self, index=0):

        test_remainder = self.test.shape[0] % self.n_output
        if test_remainder != 0:
            test = self.test[:-test_remainder]
        else:
            test = self.test

        history = [x for x in self.train[-self.n_input:, :]]
        history.extend(test)

        if index < len(test)-self.n_output:
            index = index
        else:
            index = len(test)-self.n_output

        x_input = np.array(history[index:index+self.n_input])
        x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
        # print(x_input)
        yhat_sequence = self.forcast(x_input)
        actual = test[index:index+self.n_output, 0]

        return np.array(yhat_sequence), actual

    def walk_forward_validation(self, pred_length, start_point=0):
        """
        walk-forward validation for univariate data.
        """
        test_remainder = self.test.shape[0] % self.n_output
        if test_remainder != 0:
            test = self.test[:-test_remainder]
        else:
            test = self.test

        history = [x for x in self.train[-self.n_input:, :]]
        history.extend(test)

        if start_point < len(test)-pred_length:
            start_point = start_point
        else:
            start_point = len(test)-pred_length

        inputs = np.array(history[start_point:start_point + self.n_input])
        predictions = []
        actuals = []

        max_length = len(self.test) - (start_point + 1) - self.n_output

        if pred_length > max_length:
            pred_length = max_length
        else:
            pred_length = pred_length

        step = round(pred_length / self.n_output)

        for i in range(step):
            # Prepare the input sequence
            x_input = inputs[-self.n_input:]
            x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))

            # Make prediction
            yhat_sequence = self.forcast(x_input)
            # print(yhat_sequence)
            predictions.append(yhat_sequence)

            # Get actual value for the current timestep
            actual = self.test[start_point:start_point + self.n_output, 0]
            # print(actual)
            actuals.append(actual)

            # Update the input sequence
            x_input_new = self.test[start_point:start_point + self.n_output, :]
            print(x_input_new)
            for j in range(len(yhat_sequence)):
                np.put(x_input_new[j], 0, yhat_sequence[j])
            print(x_input_new)
            inputs = np.append(inputs, x_input_new, axis=0)

            # Store the predicted value and actual value

            start_point += self.n_output

        return np.array(predictions), np.array(actuals)


class Evaluate:
    def __init__(self, actual, predictions) -> None:
        if actual.shape[1] > 1:
            actual_values = actual[:, 0]
        else:
            actual_values = actual

        self.actual = actual_values
        self.predictions = predictions
        self.var_ratio = self.compare_var()
        self.mape = self.evaluate_model_with_mape()

    def compare_var(self) -> float:
        """
        Calculates the variance ratio of the predictions
        """
        return abs(1 - (np.var(self.predictions)) / np.var(self.actual))

    def evaluate_model_with_mape(self) -> float:
        """
        Calculates the mean absolute percentage error
        """
        return mean_absolute_percentage_error(self.actual.flatten(), self.predictions.flatten())


# class SampleForcast:
#     def __init__(self, model, train, test, n_input=5, n_output=5):
#         self.model = model
#         self.train = train
#         self.test = test
#         self.n_input = n_input
#         self.n_output = n_output
#
#     def get_sample(self, index=0):


def plot_metrics(history, epochs: int = 25):
    acc = history.history['mape']
    val_acc = history.history['val_mape']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training MAPE')
    plt.plot(epochs_range, val_acc, label='Validation MAPE')
    plt.legend(loc='upper right')
    # plt.ylim(0, 50)
    plt.title('Training and Validation MAPE')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    # plt.ylim(0, 100)
    plt.title('Training and Validation Loss')
    plt.show()


def plot_results(test, preds, df, title_suffix=None, xlabel='Power Prediction'):
    """
    Plots training data in blue, actual values in red, and predictions in green, over time.
    """
    fig, ax = plt.subplots(figsize=(18, 6))
    # x = df.Close[-498:].index
    if test.shape[1] > 1:
        test = test[:, 0]

    plot_test = test[0:]
    plot_preds = preds[0:]

    # x = df[-(plot_test.shape[0] * plot_test.shape[1]):].index
    # plot_test = plot_test.reshape((plot_test.shape[0] * plot_test.shape[1], 1))
    plot_preds = plot_preds.reshape((plot_preds.shape[0] * plot_preds.shape[1], 1))

    ax.plot(plot_test, label='actual')
    ax.plot(plot_preds, label='preds')

    if title_suffix is None:
        ax.set_title('Predictions vs. Actual')
    else:
        ax.set_title(f'Predictions vs. Actual, {title_suffix}')

    ax.set_xlabel('Date')
    ax.set_ylabel(xlabel)
    ax.legend()

    plt.show()


def plot_sample_results(test, preds):
    plt.plot(test, color='black', label='Actual')
    plt.plot(preds, color='green', label='Predicted')
    # plt.title('Power Prediction')
    # plt.xlabel('Hour')
    # plt.ylabel('kWh')
    plt.legend()
    plt.ylim(0, 200)
    plt.show()
