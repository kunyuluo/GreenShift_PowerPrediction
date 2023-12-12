#%%
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import pytz

DAY_NUM = 23
MONTH = 11

#load 1 min test data and moer data
df_test = pd.read_csv('2023{}{}_RealTimeData.csv'.format(MONTH,DAY_NUM))
df_test = df_test.fillna(0)

#read point list file
# Point_list_file = 'df_statics_webctrl_v2.csv'
# df_pl = pd.read_csv(Point_list_file, index_col = 'Points Name')

df_test_day = lambda df, day_num: df[df['time_local'].dt.day == day_num]

def transfer_time_zone(df, datatime = 'data_time'):
    df['time_UTC'] = pd.to_datetime(df[datatime]).dt.tz_localize('UTC')
    df['time_local'] = df['time_UTC'].dt.tz_convert(pytz.timezone('US/Eastern')).dt.tz_localize(None).dt.floor('min')
    df['time_UTC'] = df['time_UTC'].dt.tz_localize(None)
    return df

def plotting_variables(df, ylabels, legends, ylims = None, adjust = True, label_1 = None, label_2 = None):
    # df_temp = df_test_day(df, day_num)
    colors=('b', 'r', 'g', 'c', 'm', 'y', 'k')
    fig = plt.figure(figsize=(15,3)) 
    ax = fig.add_subplot()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval = 2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval = 1))
    lns = []
    fig.subplots_adjust(right=0.8)
    if adjust == True:
        for i, y in enumerate(ylabels):
            if i == 0:
                axtemp = ax
                # axtemp.set_ylim(ylims[i])
            else:
                axtemp = ax.twinx()
                if i > 1:
                    axtemp.spines.right.set_position(("axes", 1+i/15))
            ln = axtemp.plot(df.index, df[y], color = colors[i],label = legends[i])
            lns = lns + ln
            axtemp.set_ylabel(ylabel =legends[i])
            # axtemp.set_ylim(ylims[i])
    else:
        ax1 = ax.twinx()
        for i, y in enumerate(ylabels):
            if i <= 1:
                axtemp = ax
                axtemp.set_ylabel(label_1)
                # axtemp.set_ylim(ylims[0])
            else:
                axtemp = ax1
                axtemp.set_ylim(ylims[1])
                axtemp.set_ylabel(label_2)
            ln = axtemp.plot(df.index, df[y], color = colors[i],label = legends[i])
            lns = lns + ln
            # axtemp.set_ylim(ylims[i])
        # axtemp.set_ylabel(label_2)

    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=4)
    plt.show()

    return fig

#%%
# transfer time zone from UTC to local time
df_test = transfer_time_zone(df_test)

#%%
# merge test data with moer data to be df in the selected day
df_test = df_test_day(df_test, DAY_NUM)
df_test.set_index('time_local',inplace=True)
# df_test = df_test.resample('5T',axis=0).asfreq()
df_test = df_test[df_test.index.hour.isin(range(8,22))]
df_test.reset_index(inplace=True)


#%%
df_test.set_index('time_local',inplace=True)
df_test = df_test[df_test.index.hour.isin(range(8,20))]

#%%
vars_3 = ['executive_suite_temp']
legends_3 = ['executive suite zone temp[F]']
ylims_3 = [(70,80)]
fig3 = plotting_variables(df_test, vars_3, legends_3, ylims_3,False,'Zone Temp [F]')
