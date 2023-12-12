import pandas as pd
import numpy as np
from Helper import format_date, transfer_time_zone
from Helper import select_by_date, select_by_time, minute_average_by_day
from Helper import plot_variable, plot_variable_no_time

# Load data
# *******************************************************************************
data = pd.read_csv('gs_real_time_data.csv', low_memory=False)

# Time zone transfer from UTC to local ('US/Eastern)
# *******************************************************************************
date_local = transfer_time_zone(data)
# date_local = format_date(data)

# Get data from specific column
# *******************************************************************************
power_data = pd.concat([date_local, data['cp_power']], axis=1)
print(power_data)

target = power_data[(power_data['data_time'].dt.month == 8) &
                    (power_data['data_time'].dt.day == 18) &
                    (power_data['data_time'].dt.hour == 3) &
                    (power_data['data_time'].dt.minute == 22)]['cp_power'].values
print(target)
# day = power_data[((power_data['weekday'] == 3) |
#                  (power_data['weekday'] == 4)) &
#                  (power_data['data_time'].dt.hour == 10) &
#                  (power_data['data_time'].dt.minute == 0)]

# default = minute_average_by_day(power_data, 'cp_power')
# print(default[0][11])

# Get data for specified period
# *******************************************************************************
# start_month, start_day, end_month, end_day = 9, 1, 10, 19
# power_period = select_by_date(power_data, start_month, start_day, end_month, end_day)
# power_period = select_by_time(power_period, 8, 20)
# # power_period = power_period.groupby(pd.Grouper(freq='5min')).mean()
# power_period = power_period.dropna()

# print(power_period)

# Plot the selected data
# *******************************************************************************
# plot_variable(power_period, 'cp_power', False)
# plot_variable_no_time(power_period, 'cp_power')
