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


# def fill_missing_value(df: pd.DataFrame, column_name: str = 'cp_power'):



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


def plot_variable_no_time(df: pd.DataFrame, var_name: str = 'cp_power'):
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot()

    x = range(len(df[var_name]))
    ax.plot(x, df[var_name], color='black')
    ax.set_ylim(0, )

    plt.show()
