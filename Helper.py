import numpy as np
import pandas as pd
import pytz
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries
from darts.models import TiDEModel
from Time_Features import TimeCovariates


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
        # self.new_dataset = self.fill_missing_value()

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

        data = self.get_feature_data()
        # Construct new dataframe without missing value for each timestep
        # *******************************************************************************
        datetimes = []
        dt_min = data['data_time'].min()
        dt_max = data['data_time'].max()
        year_range = range(dt_min.year, dt_max.year + 1)

        months_range = {}
        days_range = {}
        for year in year_range:
            month_min = data[data['data_time'].dt.year == year]['data_time'].dt.month.min()
            month_max = data[data['data_time'].dt.year == year]['data_time'].dt.month.max()
            day_min = data[(data['data_time'].dt.year == year) &
                           (data['data_time'].dt.month == month_min)]['data_time'].dt.day.min()
            day_max = data[(data['data_time'].dt.year == year) &
                           (data['data_time'].dt.month == month_max)]['data_time'].dt.day.max()
            months_range[year] = range(month_min, month_max + 1)
            days_range[year] = (day_min, day_max)

        # start_day, end_day = dt_min.day, dt_max.day
        num_days_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        for year in year_range:
            month_range = months_range[year]
            if len(months_range[year]) == 1:
                for day in range(1, days_range[year][1] + 1):
                    for hour in range(24):
                        for minute in range(60):
                            datetimes.append(
                                dt.datetime(year=year, month=month_range[0], day=day, hour=hour, minute=minute))
            else:
                for i, month in enumerate(month_range):
                    if i == 0:
                        for day in range(days_range[year][0], num_days_month[month - 1] + 1):
                            for hour in range(24):
                                for minute in range(60):
                                    datetimes.append(
                                        dt.datetime(year=year, month=month, day=day, hour=hour, minute=minute))
                    elif i == len(month_range) - 1:
                        for day in range(1, days_range[year][1] + 1):
                            for hour in range(24):
                                for minute in range(60):
                                    datetimes.append(
                                        dt.datetime(year=year, month=month, day=day, hour=hour, minute=minute))
                    else:
                        for day in range(1, num_days_month[i] + 1):
                            for hour in range(24):
                                for minute in range(60):
                                    datetimes.append(
                                        dt.datetime(year=year, month=month, day=day, hour=hour, minute=minute))

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
            elif data[i] != 0 and data[i - 1] != 0 and 1 - (data[i] / data[i - 1]) > 0.8:
                data[i] = data[i - 1] * 0.5
            elif data[i] != 0 and data[i + 1] != 0 and 1 - (data[i + 1] / data[i]) > 0.8:
                data[i] = data[i + 1] * 0.5
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
                 start_date: tuple = None,
                 end_date: tuple = None,
                 hour_range: tuple = None,
                 group_freq: int = None,
                 test_size: float = 0.2,
                 n_input: int = 5,
                 n_output: int = 5,
                 time_zone_transfer: bool = False,
                 date_column: str = 'data_time',
                 scaler=None):

        feature_names = [] if feature_names is None else feature_names

        data = df if df is not None else pd.read_csv(file_path, low_memory=False)

        num_features = len(feature_names) if len(feature_names) > 0 else data.shape[1] - 1

        self.df = data
        self.feature_names = feature_names
        self.start_date = start_date
        self.end_date = end_date
        self.hour_range = hour_range
        self.group_freq = group_freq
        self.test_size = test_size
        self.n_input = n_input
        self.n_output = n_output
        self.time_zone_transfer = time_zone_transfer
        self.date_column = date_column
        self.scaler = scaler
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
            # df_selected = df[
            #     (df[self.date_column].dt.year >= start_year) &
            #     (df[self.date_column].dt.year <= end_year) &
            #     (df[self.date_column].dt.month >= start_month) &
            #     (df[self.date_column].dt.month <= end_month) &
            #     (df[self.date_column].dt.day >= start_day) &
            #     (df[self.date_column].dt.day <= end_day)]

            start = pd.to_datetime(dt.datetime(start_year, start_month, start_day))
            end = pd.to_datetime(dt.datetime(end_year, end_month, end_day))

            df_selected = df[(df[self.date_column] >= start) & (df[self.date_column] < end)]

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
        if self.start_date is not None and len(self.start_date) == 3:
            target_period = self.select_by_date(
                target_data, start_year=self.start_date[0], end_year=self.end_date[0],
                start_month=self.start_date[1], start_day=self.start_date[2],
                end_month=self.end_date[1], end_day=self.end_date[2])
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
        # return target_data

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
        # print(data.index)

        if self.scaler is not None:
            index = data.index
            column_names = data.columns
            data = self.scaler.fit_transform(data)
            data = pd.DataFrame(data, index=index, columns=column_names)
            # print(data.head(30))

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
        # print(train_remainder, test_remainder)

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


class DataPreprocessorTiDE:
    def __init__(self,
                 file_path: str,
                 df: pd.DataFrame = None,
                 target_names=None,
                 dynamic_cov_names=None,
                 static_cov_names=None,
                 start_date: tuple = None,
                 end_date: tuple = None,
                 hour_range: tuple = None,
                 group_freq: int = None,
                 test_size: float = 0.2,
                 val_size: float = 0.05,
                 time_zone_transfer: bool = False,
                 date_column: str = 'data_time',
                 add_time_features: bool = False,
                 scaler=None):

        target_names = [] if target_names is None else target_names
        dynamic_cov_names = [] if dynamic_cov_names is None else dynamic_cov_names
        # if dynamic_cov_names is None:
        #     dynamic_cov_names = []
        # else:
        #     if add_time_features:
        #         dynamic_cov_names = dynamic_cov_names +
        static_cov_names = [] if static_cov_names is None else static_cov_names

        data = df if df is not None else pd.read_csv(file_path, low_memory=False)

        self.df = data
        self.target_names = target_names
        self.dynamic_cov_names = dynamic_cov_names
        self.static_cov_names = static_cov_names
        self.start_date = start_date
        self.end_date = end_date
        self.hour_range = hour_range
        self.group_freq = group_freq
        self.time_zone_transfer = time_zone_transfer
        self.date_column = date_column
        self.add_time_features = add_time_features
        self.scaler = scaler
        self.all_features = target_names + dynamic_cov_names + static_cov_names

        period_data = self.get_period_data()
        self.data = period_data[0]

        self.train_idx = round(len(self.data) * (1 - test_size))
        self.val_idx = round(len(self.data) * (1 - val_size)) if val_size < 1 else len(self.data) - val_size

        self.time_feature_names = period_data[1]

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
            start = pd.to_datetime(dt.datetime(start_year, start_month, start_day))
            end = pd.to_datetime(dt.datetime(end_year, end_month, end_day))

            df_selected = df[(df[self.date_column] >= start) & (df[self.date_column] < end)]

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
        # print(date_local)
        # Get data from specific column
        # *******************************************************************************
        if len(self.all_features) != 0:
            target_data = pd.concat([date_local, self.df[self.all_features]], axis=1)
        else:
            feature_df = self.df.drop([self.date_column], axis=1)
            target_data = pd.concat([date_local, feature_df], axis=1)

        # Get data for specified period
        # *******************************************************************************
        if self.start_date is not None and len(self.start_date) == 3:
            target_period = self.select_by_date(
                target_data, start_year=self.start_date[0], end_year=self.end_date[0],
                start_month=self.start_date[1], start_day=self.start_date[2],
                end_month=self.end_date[1], end_day=self.end_date[2])
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

        if self.scaler is not None:
            index = target_period.index
            column_names = target_period.columns
            target_period = self.scaler.fit_transform(target_period)
            target_period = pd.DataFrame(target_period, index=index, columns=column_names)

        time_feas = []

        # Add time features as dynamic covariates if needed:
        if self.add_time_features:
            dti = target_period.index
            time_covs = TimeCovariates(dti)
            time_features = time_covs.get_covariates()
            target_period = pd.concat([target_period, time_features], axis=1)

            time_feas = time_covs.get_feature_names()

        return target_period, time_feas

    def split_data(self):
        """
        Split data into train, test, and validation sets.
        """
        if len(self.data) != 0:
            train = self.data[:self.train_idx]
            test = self.data[self.train_idx: self.val_idx]
            val = self.data[self.val_idx:]

            return train, test, val
        else:
            raise Exception('Data set is empty, cannot split.')

    def train_series(self):

        train, test, val = self.split_data()

        past_cov_names = self.dynamic_cov_names + self.time_feature_names

        train_target = TimeSeries.from_dataframe(
            train[self.target_names], fill_missing_dates=True, fillna_value=0.0, freq=f'{self.group_freq}min')
        train_past_covs = TimeSeries.from_dataframe(
            train[past_cov_names], fill_missing_dates=True, fillna_value=0.0, freq=f'{self.group_freq}min')

        return train_target, train_past_covs

    def test_series(self):
        # data = self.get_period_data()
        train, test, val = self.split_data()

        past_cov_names = self.dynamic_cov_names + self.time_feature_names

        test_target = TimeSeries.from_dataframe(
            test[self.target_names], fill_missing_dates=True, fillna_value=0.0, freq=f'{self.group_freq}min')
        test_past_covs = TimeSeries.from_dataframe(
            test[past_cov_names], fill_missing_dates=True, fillna_value=0.0, freq=f'{self.group_freq}min')

        return test_target, test_past_covs

    def val_series(self):
        # data = self.get_period_data()
        train, test, val = self.split_data()

        past_cov_names = self.dynamic_cov_names + self.time_feature_names

        val_target = TimeSeries.from_dataframe(
            val[self.target_names], fill_missing_dates=True, fillna_value=0.0, freq=f'{self.group_freq}min')
        val_past_covs = TimeSeries.from_dataframe(
            val[past_cov_names], fill_missing_dates=True, fillna_value=0.0, freq=f'{self.group_freq}min')

        return val_target, val_past_covs

    def future_series(self, cov_names, n_extend):
        """
        Build future covariates for the given features

        Span of future covariate:
        with n <= output_chunk_length, at least the same time span as target plus the next output_chunk_length time steps after the end of target.
        with n > output_chunk_length, at least the same time span as target plus the next n time steps after the end of target.
        (source: https://github.com/unit8co/darts/blob/master/docs/userguide/covariates.md)
        """
        n_extend = n_extend if n_extend > 24 else 24
        future_length = n_extend + self.train_idx

        if isinstance(cov_names, str):
            cov_names = [cov_names]
            future_covs = self.data[cov_names]
        elif isinstance(cov_names, list):
            future_covs = self.data[cov_names]
        else:
            raise Exception('Invalid cov_names')

        future_covs = future_covs[:future_length]
        future_covs = TimeSeries.from_dataframe(future_covs, fill_missing_dates=True)

        return future_covs


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

        if index < len(test) - self.n_output:
            index = index
        else:
            index = len(test) - self.n_output

        x_input = np.array(history[index:index + self.n_input])
        x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
        # print(x_input)
        yhat_sequence = self.forcast(x_input)
        actual = test[index:index + self.n_output, 0]

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

        if start_point < len(test) - pred_length:
            start_point = start_point
        else:
            start_point = len(test) - pred_length

        inputs = np.array(history[start_point:start_point + self.n_input])
        predictions = []
        actuals = []

        max_length = len(test) - (start_point + 1) - self.n_output

        if pred_length > max_length:
            pred_length = max_length
        else:
            pred_length = pred_length

        step = round(pred_length / self.n_output)

        for i in range(step):
            # Prepare the input sequence
            x_input = inputs[-self.n_input:]
            # print(x_input)
            x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))

            # Make prediction
            yhat_sequence = self.forcast(x_input)
            for value in yhat_sequence:
                # print(yhat_sequence)
                predictions.append(value)

            # Get actual value for the current timestep
            actual = test[start_point:start_point + self.n_output, 0]
            for value in actual:
                # print(actual)
                actuals.append(value)

            # Update the input sequence
            x_input_new = test[start_point:start_point + self.n_output, :]
            # print(x_input_new)

            for j in range(len(yhat_sequence)):
                # np.put(x_input_new[j], 0, yhat_sequence[j])
                x_input_new[j, 0] = yhat_sequence[j]

            inputs = np.append(inputs, x_input_new, axis=0)

            start_point += self.n_output

        return np.array(predictions).reshape(-1, 1), np.array(actuals).reshape(-1, 1)


class PredictAndForecastTiDE:
    """
        model: Darts TiDE Model
        target: TimeSeries of features to be predicted
        past_covs: TimeSeries of past covariates
        future_covs: TimeSeries of future covariates

    """

    def __init__(
            self,
            model: TiDEModel,
            input_chunk_length,
            output_chunk_length,
            target=None,
            past_covs=None,
            future_covs=None) -> None:

        self.model = model
        self.n_input = input_chunk_length
        self.n_output = output_chunk_length
        self.target = target
        self.past_covs = past_covs
        self.future_covs = future_covs

    def forcast(self, series=None, past_covs=None, future_covs=None):
        yhat = self.model.predict(
            n=self.n_output, series=series, past_covariates=past_covs, future_covariates=future_covs)

        # we only want the vector forecast
        yhat = yhat.pd_dataframe().values
        return yhat

    # def get_predictions(self):
    #     # history is a list of flattened test data + last observation from train data
    #     test_remainder = self.test.shape[0] % self.n_output
    #     if test_remainder != 0:
    #         test = self.test[:-test_remainder]
    #     else:
    #         test = self.test
    #
    #     history = [x for x in self.train[-self.n_input:, :]]
    #     history.extend(test)
    #
    #     step = round(len(history) / self.n_output)
    #     # history = []
    #
    #     # walk-forward validation
    #     predictions = []
    #     window_start = 0
    #     for i in range(step):
    #
    #         if window_start <= len(history) - self.n_input - self.n_output:
    #             # print('pred no {}, window_start {}'.format(i+1, window_start))
    #             x_input = np.array(history[window_start:window_start + self.n_input])
    #             x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
    #             yhat_sequence = self.forcast(x_input)
    #             # print('pred no {}'.format(i))
    #             # store the predictions
    #             predictions.append(yhat_sequence)
    #
    #         window_start += self.n_output
    #         # get real observation and add to history for predicting the next week
    #         # history.append(self.test[i, :])
    #
    #     return np.array(predictions)


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
        # print(np.var(self.predictions))
        # print(np.var(self.actual))
        return abs(1 - (np.var(self.predictions)) / np.var(self.actual))

    def evaluate_model_with_mape(self) -> float:
        """
        Calculates the mean absolute percentage error
        """
        return mean_absolute_percentage_error(self.actual.flatten(), self.predictions.flatten())


def inverse_transform_prediction(values, feature_num: int, scaler):
    values = values.flatten()

    dim = values.shape[0]
    dummy_array = np.zeros([dim, feature_num])
    for i in range(dim):
        np.put(dummy_array[i], 0, values[i])

    unscaled = scaler.inverse_transform(dummy_array)
    unscaled = unscaled[:, 0]
    unscaled = unscaled.reshape((unscaled.shape[0], 1))

    return unscaled


def scale_data(value, value_range=(0, 1), scaled_range=(0, 1)):
    ratio = (value - value_range[0]) / (value_range[1] - value_range[0])
    return ratio * (scaled_range[1] - scaled_range[0]) + scaled_range[0]
