import numpy as np
import pandas as pd
import pytz
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def format_date(df: pd.DataFrame, column: str = 'data_time'):
    df['data_time'] = pd.to_datetime(df[column])

    return df


def transfer_time_zone(df: pd.DataFrame, column: str = 'data_time'):
    df['data_time'] = pd.to_datetime(df[column])
    df_utc = df['data_time'].dt.tz_localize('UTC')
    df_local = df_utc.dt.tz_convert(pytz.timezone('US/Eastern')).dt.tz_localize(None).dt.floor('min')
    # df_utc = df_utc.dt.tz_localize(None)

    return df_local


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


def select_by_date(
        df: pd.DataFrame,
        start_month: int = 10,
        start_day: int = 16,
        end_month: int = 10,
        end_day: int = 16):
    year = dt.date.today().year
    date_start = dt.date(year, start_month, start_day)
    date_end = dt.date(year, end_month, end_day)
    days = (date_end - date_start).days + 1

    df_selected = df[
        (df['data_time'].dt.month >= start_month) &
        (df['data_time'].dt.month <= end_month) &
        (df['data_time'].dt.day >= start_day) &
        (df['data_time'].dt.day <= end_day)]

    df_selected.set_index('data_time', inplace=True)

    return df_selected


def select_by_time(df: pd.DataFrame, start_hour: int = 8, end_hour: int = 22):
    df_selected = df[
        (df.index.hour >= start_hour) &
        (df.index.hour < end_hour)]

    return df_selected


def minute_average_by_day(df: pd.DataFrame, column_name: str = 'cp_power'):
    """
    Calculate average value of every minute in a day by weekday (from Monday to Sunday).
    Use the calculated value to fill empty/missing value in the dataset.
    """
    # try:
    #     df.set_index('data_time', inplace=True)
    # except IndexError:
    #     pass

    df['weekday'] = df['data_time'].dt.weekday

    hours = range(24)
    minutes = range(60)
    default_values = {}

    weekdays = {}
    weekends = {}
    for hour in hours:
        hours_wday = []
        hours_wend = []
        for minute in minutes:
            value_wday = df[((df['weekday'] == 0) |
                             (df['weekday'] == 1) |
                             (df['weekday'] == 2) |
                             (df['weekday'] == 3) |
                             (df['weekday'] == 4)) &
                            (df['data_time'].dt.hour == hour) &
                            (df['data_time'].dt.minute == minute)][column_name].mean()

            value_wend = df[((df['weekday'] == 5) |
                             (df['weekday'] == 6)) &
                            (df['data_time'].dt.hour == hour) &
                            (df['data_time'].dt.minute == minute)][column_name].mean()
            hours_wday.append(value_wday)
            hours_wend.append(value_wend)
        weekdays[hour] = hours_wday
        weekends[hour] = hours_wend

    default_values[0] = weekdays
    default_values[1] = weekends

    return default_values


class DefaultValueFiller:
    def __init__(self, df: pd.DataFrame, feature_names=None):

        feature_names = [] if feature_names is None else feature_names

        self.df = df
        self.feature_names = feature_names
        self.datetime_column = 'data_time'
        self.feature_data = self.get_feature_data()
        self.new_dataset = self.fill_missing_value()

    def get_feature_data(self):
        # Time zone transfer from UTC to local ('US/Eastern)
        # *******************************************************************************
        date_local = transfer_time_zone(self.df)

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

            new_df[feature] = filled_data
            # new_df.drop(['weekday'], axis=1, inplace=True)

        return new_df

    def calc_default_value(self, column_name: str = 'cp_power'):
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
    def __init__(self, data_path: str, target_column=None):
        target_column = ['cp_power'] if target_column is None else target_column

        self.data_path = data_path
        self.df = pd.read_csv(data_path, low_memory=False)
        self.target_column = target_column

    def format_date(self, column: str = 'data_time'):
        self.df['data_time'] = pd.to_datetime(self.df[column])

        return self.df

    def transfer_time_zone(self, column: str = 'data_time'):
        self.df['data_time'] = pd.to_datetime(self.df[column])
        df_utc = self.df['data_time'].dt.tz_localize('UTC')
        df_local = df_utc.dt.tz_convert(pytz.timezone('US/Eastern')).dt.tz_localize(None).dt.floor('min')
        # df_utc = df_utc.dt.tz_localize(None)

        return df_local

    def fill_missing_value(self):
        """
        Fill missing value in the dataset.
        """
        target_column = self.target_column

        if len(target_column) != 0:
            for column in target_column:
                self.df[column] = self.df[column].fillna(method='ffill')

        return self.df

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
