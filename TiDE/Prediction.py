from Helper import DataPreprocessorTiDE
from Helper import plot_results
from darts.models import TiDEModel
from sklearn.preprocessing import MinMaxScaler
from darts.metrics import mape

# Load the model
# *******************************************************************************
model = TiDEModel.load('models/model_tide.pkl')

# Get data from specific column
# *******************************************************************************
file_path = '../new_data_0102.csv'
target_names = ['cp_power']
dynamic_cov_names = ['oat', 'oah', 'downstream_chwsstpt']
sc = MinMaxScaler(feature_range=(0, 1))

data_loader = DataPreprocessorTiDE(
    file_path,
    target_names=target_names,
    dynamic_cov_names=dynamic_cov_names,
    # start_date=(2023, 12, 31),
    # end_date=(2024, 1, 2),
    hour_range=(6, 20),
    group_freq=15,
    test_size=0.2,
    val_size=24,
    scaler=sc
)

train_target_series, train_past_covs = data_loader.train_series()
test_target_series, test_past_covs = data_loader.test_series()

# future_covs = dataset.future_series(cov_names, n_extend) if use_future_covs else None
# Predict for the entire test set:
# *******************************************************************************
pred = model.predict(
    n=12,
    series=train_target_series,
    past_covariates=train_past_covs,
    num_loader_workers=11)

print(pred.pd_dataframe())

# Evaluate the prediction
# *******************************************************************************
mape = mape(test_target_series, pred)
print('TiDE Model\'s mape is: {}'.format(mape))

# Visualize the results
# *******************************************************************************
# plot_results()
