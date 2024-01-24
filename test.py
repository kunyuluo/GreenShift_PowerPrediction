import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
from Helper import DefaultValueFiller
from Helper import GSDataProcessor
from Helper import inverse_transform_prediction, scale_data
from sklearn.preprocessing import MinMaxScaler
from Visualization import VisualizeData

# Load data
# *******************************************************************************
# data = pd.read_csv('gs_real_time_data_new.csv', low_memory=False)
# print(data)

# Time zone transfer from UTC to local ('US/Eastern)
# *******************************************************************************
# date_local = transfer_time_zone(data)
# date_local = format_date(data)
# print(date_local)

# Get data from specific column
# *******************************************************************************
features_name = ['cp_power', 'oat', 'oah', 'downstream_chwsstpt']

# filler = DefaultValueFiller(data, features_name)
# feature_data = filler.get_feature_data()
# new_df = filler.fill_missing_value()
# print(new_df)
# new_df.to_csv('new_data_0123.csv', index=False)

# power_data = pd.concat([date_local, data[features_name]], axis=1)
# power_data['weekday'] = power_data['data_time'].dt.weekday
# print(power_data)

sc = MinMaxScaler(feature_range=(0, 1))

target_data = GSDataProcessor(
    'Data/new_data_0102.csv',
    feature_names=features_name,
    start_date=(2023, 9, 18),
    end_date=(2023, 9, 20),
    # hour_range=(6, 20),
    group_freq=15,
    n_input=6,
    n_output=6,)
    # scaler=sc)

period_data = target_data.get_period_data()
# print(period_data)
#
# train = target_data.train
# train = train.reshape(train.shape[0] * train.shape[1], train.shape[2])
# test = target_data.test
# test = test.reshape(test.shape[0] * test.shape[1], test.shape[2])
# print(test)
# print(train.shape)
# spt = train[:, 3]
# spt = sc.fit_transform(spt.reshape(-1, 1))
# print(spt)
# print(scale_data(44, (41, 49)))
# df = pd.DataFrame(spt)
# df.to_csv('spt.csv')

# arr1 = np.array([[60, 2, 3, 44], [70, 3, 4, 46], [80, 4, 5, 48], [100, 5, 6, 44]])
# df = pd.DataFrame(arr1)
# print(df)
# arr2 = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
# data = sc.fit_transform(df)
# print(data)
# test = np.array([[0.333, 0, 0, 0], [0.333, 0, 0, 0]])
# test = sc.inverse_transform(test)
# test = np.array([0.333333, 0.333333])
# test = inverse_transform_prediction(test, len(features_name), sc)
# print(test)

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
# VisualizeData.plot_variable(period_data, 'cp_power', False)
VisualizeData.plot_variable_no_time(period_data, 'cp_power')
# VisualizeData.check_data_distribution(data, 'downstream_chwsstpt')
# VisualizeData.check_linearity(period_data, 'cp_power', 'downstream_chwsstpt', True)
# VisualizeData.check_autocorrelation(data, 'cp_power')
