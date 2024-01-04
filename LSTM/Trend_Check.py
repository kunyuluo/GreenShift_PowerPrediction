import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

data = pd.read_csv('trend_test_0104.csv')

spt = 46
power_delta = 'power_delta_{}'.format(str(spt))
spt_delta = 'spt_delta_{}'.format(str(spt))
data = data[[power_delta, spt_delta]].astype(float)

# scaler1 = MinMaxScaler(feature_range=(-1, 1))
# data = scaler1.fit_transform(data)
# data = pd.DataFrame(data, columns=['power_delta','spt_delta'])

trend_data = data[((data[power_delta] > 0) & (data[spt_delta] <= 0)) |
                  ((data[power_delta] < 0) & (data[spt_delta] >= 0))]

untrend_data = data[((data[power_delta] > 0) & (data[spt_delta] > 0)) |
                    ((data[power_delta] < 0) & (data[spt_delta] < 0))]

rate = len(trend_data) / len(data)
print('Correction rate is {}%'.format(round(rate*100, 2)))

plt.scatter(trend_data[spt_delta], trend_data[power_delta], c='green', s=4)
plt.scatter(untrend_data[spt_delta], untrend_data[power_delta], c='red', s=4)
plt.show()
