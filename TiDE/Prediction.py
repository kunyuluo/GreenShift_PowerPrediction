import pandas as pd
from Helper import DataPreprocessorTiDE
from Visualization import VisualizeData
from darts.models import TiDEModel
from sklearn.preprocessing import MinMaxScaler
from darts.metrics import mape
from darts import concatenate
from darts import TimeSeries
from Helper import inverse_transform_prediction
from Helper import Evaluate

# Load the model
# *******************************************************************************
model = TiDEModel.load('models/model_tide.pkl')

# Get data from specific column
# *******************************************************************************
file_path = '../Data/new_data_0102.csv'
target_names = ['cp_power']
dynamic_cov_names = ['oat', 'oah', 'downstream_chwsstpt']
sc = MinMaxScaler(feature_range=(0, 1))

data_loader = DataPreprocessorTiDE(
    file_path,
    target_names=target_names,
    dynamic_cov_names=dynamic_cov_names,
    # start_date=(2023, 12, 31),
    # end_date=(2024, 1, 2),
    # hour_range=(6, 20),
    group_freq=15,
    test_size=0.2,
    val_size=44,
    # add_time_features=True,
    scaler=sc
)

# train, test, val = data_loader.split_data()
# print(val)
train_target_series, train_past_covs = data_loader.train_series()
test_target_series, test_past_covs = data_loader.test_series()
val_target_series, val_past_covs = data_loader.val_series()
merged_series = concatenate([train_target_series, test_target_series], axis=0, ignore_time_axis=True)
merged_past_covs = concatenate([train_past_covs, test_past_covs], axis=0, ignore_time_axis=True)
# print(train_past_covs.pd_dataframe())
# print(test_target_series.pd_dataframe())
# val_target_series = val_target_series.drop_after(pd.Timestamp('2024-01-01T20'))
print(val_target_series.pd_dataframe().head(10))
# series = TimeSeries.from_dataframe(val_target_series.pd_dataframe(), freq='15min')
# print(series.pd_dataframe())
# print(merged_series.pd_dataframe())
# print(merged_past_covs.pd_dataframe())

# future_covs = dataset.future_series(cov_names, n_extend) if use_future_covs else None
# Predict for the entire test set:
# *******************************************************************************
pred = model.predict(
    n=10,
    series=merged_series,
    past_covariates=merged_past_covs)

# print(pred.pd_dataframe())
#
actuals = inverse_transform_prediction(val_target_series[:10].pd_dataframe().values, 4, sc)
preds = inverse_transform_prediction(pred.pd_dataframe().values, 4, sc)

# Evaluate the prediction
# *******************************************************************************
evals = Evaluate(actuals, preds)
print('TiDE Model\'s mape is: {}%'.format(round(evals.mape*100, 1)))

# Visualize the results
# *******************************************************************************
VisualizeData.plot_results(actuals, preds, ylim=(0, 100))
