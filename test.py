import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
from Helper import DefaultValueFiller
from Helper import GSDataProcessor
from sklearn.preprocessing import MinMaxScaler

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
features_name = ['cp_power', 'oat', 'oah', 'downstream_chwsstpt']

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
    start_month=8,
    start_day=18,
    end_month=11,
    end_day=11,
    hour_range=(8, 20),
    group_freq=5,
    n_input=24,
    n_output=6)

# df = target_data.get_period_data()
# print(df)

train = target_data.train
train = train.reshape(train.shape[0] * train.shape[1], train.shape[2])
test = target_data.test
test = test.reshape(test.shape[0] * test.shape[1], test.shape[2])

# spt = test[:, 3]
# print(spt)
df = pd.DataFrame(test)
df.to_csv('test_trimmed.csv')
# n_input = 7
# n_output = 1
# start_point = 0
# pred_length = 10
#
# test_remainder = test.shape[0] % n_output
# if test_remainder != 0:
#     test = test[:-test_remainder]
# else:
#     test = test
#
# history = [x for x in train[-n_input:, :]]
# history.extend(test)

# x_input = np.array(history[0:7])
# x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
# print(x_input)
# print(x_input.shape)

# if start_point < len(test) - pred_length:
#     start_point = start_point
# else:
#     start_point = len(test) - pred_length
#
# inputs = np.array(history[start_point:start_point + n_input])

# x_input = inputs[-7:]
# print(inputs)
# print(x_input)
# print(type(x_input))

# arr1 = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
# arr2 = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
# new_values = np.array([20, 30, 40, 50])
# new_values = new_values.reshape(new_values.shape[0], 1)

# np.put(arr, [0, 0, 0, 0], new_values)
# for i in range(len(new_values)):
#     np.put(arr2[i], 0, new_values[i])
#
# arr1 = np.append(arr1, arr2, axis=0)
# print(arr1)

# Normalizing data, scale between 0 and 1:
# sc = MinMaxScaler(feature_range=(0, 1))
# train_scaled = sc.fit_transform(train)

# print(train)
# print(train.shape)
# print(train_scaled)
# print(train_scaled.shape)

# GSDataProcessor.plot_variable_no_time(preview_data, 'cp_power')

# Plot the selected data
# *******************************************************************************
# GSDataProcessor.plot_variable(power_period, 'cp_power', True)
# GSDataProcessor.plot_variable_no_time(power_period, 'cp_power')
# GSDataProcessor.check_data_distribution(data, 'downstream_chwsstpt')
# GSDataProcessor.check_linearity(preview_data, 'cp_power', 'oat', True)
# GSDataProcessor.check_autocorrelation(data, 'cp_power')
