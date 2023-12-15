import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
from Helper import DefaultValueFiller
from Helper import GSDataProcessor

# Load data
# *******************************************************************************
# data = pd.read_csv('gs_real_time_data.csv', low_memory=False)

# Time zone transfer from UTC to local ('US/Eastern)
# *******************************************************************************
# date_local = transfer_time_zone(data)
# date_local = format_date(data)
# print(date_local)

# Get data from specific column
# *******************************************************************************
features_name = ['cp_power', 'oat', 'oah']

# filler = DefaultValueFiller(data, features_name)
# new_df = filler.fill_missing_value()
# print(new_df)
# new_df.to_csv('new_data.csv', index=False)

# power_data = pd.concat([date_local, data[features_name]], axis=1)
# power_data['weekday'] = power_data['data_time'].dt.weekday
# print(power_data)

target_data = GSDataProcessor(
    'new_data.csv',
    feature_names=features_name,
    # start_month=10,
    # start_day=16,
    # end_month=10,
    # end_day=22,
    hour_range=(8, 20),
    group_freq=5).X_train

target_data = target_data.reshape((target_data.shape[0] * target_data.shape[1], target_data.shape[2]))
sample = target_data[-2:, :]
# sample = sample.reshape((1, sample.shape[0], sample.shape[1]))
print(sample)
print(sample.shape)
# print(target_data)
# print(target_data.shape)

# GSDataProcessor.plot_variable_no_time(target_data, 'cp_power')


# Construct new dataframe without missing value for each timestep
# *******************************************************************************
# datetimes = []
# dt_min = power_data['data_time'].min()
# dt_max = power_data['data_time'].max()
# year_range = range(dt_min.year, dt_max.year + 1)
# month_range = range(dt_min.month, dt_max.month + 1)
# start_day, end_day = dt_min.day, dt_max.day
# num_days_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#
# for year in year_range:
#     for i, month in enumerate(month_range):
#         if i == 0:
#             for day in range(start_day, num_days_month[month - 1] + 1):
#                 for hour in range(24):
#                     for minute in range(60):
#                         datetimes.append(dt.datetime(year=year, month=month, day=day, hour=hour, minute=minute))
#         elif i == len(month_range) - 1:
#             for day in range(1, end_day + 1):
#                 for hour in range(24):
#                     for minute in range(60):
#                         datetimes.append(dt.datetime(year=year, month=month, day=day, hour=hour, minute=minute))
#         else:
#             for day in range(1, num_days_month[i] + 1):
#                 for hour in range(24):
#                     for minute in range(60):
#                         datetimes.append(dt.datetime(year=year, month=month, day=day, hour=hour, minute=minute))
# # print(pd.to_datetime(datetimes))
# df = pd.DataFrame(pd.to_datetime(datetimes), columns=['data_time'])
# df['weekday'] = df['data_time'].dt.weekday
#
# for feature in features_name:
#     default = minute_average_by_day(power_data, feature)
#     feature_data = []
#     for date in datetimes:
#         value = power_data[(power_data['data_time'] == date)]['cp_power'].values
#
#         if len(value) == 0:
#             weekday = date.weekday()
#             if weekday in [0, 1, 2, 3, 4]:
#                 value = default[0][date.hour][date.minute]
#             elif weekday in [5, 6]:
#                 value = default[1][date.hour][date.minute]
#
#             feature_data.append(value)
#         else:
#             feature_data.append(value[0])
#
#     df[feature] = feature_data
#
# print(df)

# default = minute_average_by_day(power_data, 'cp_power')
# values_1 = []
# values_2 = []
# for hour in range(24):
#     values_1.extend(default[0][hour])
#     values_2.extend(default[1][hour])
# plt.plot(values_1, label='weekday')
# plt.plot(values_2, label='weekend')
# # plt.ylim([0, 250])
# plt.legend()
# plt.show()

# Get data for specified period
# *******************************************************************************
# start_month, start_day, end_month, end_day = 10, 16, 10, 19
# power_period = select_by_date(power_data, start_month, start_day, end_month, end_day)
# power_period = select_by_time(power_period, 8, 20)
# print(power_period)
# power_period = power_period.groupby(pd.Grouper(freq='5min')).mean()
# power_period = power_period.dropna()

# print(power_period)

# Plot the selected data
# *******************************************************************************
# GSDataProcessor.plot_variable(power_period, 'cp_power', True)
# GSDataProcessor.plot_variable_no_time(power_period, 'cp_power')
# GSDataProcessor.check_data_distribution(data, 'downstream_chwsstpt')
# GSDataProcessor.check_linearity(data, 'cp_power', 'downstream_chwsstpt', True)
# GSDataProcessor.check_autocorrelation(data, 'cp_power')
