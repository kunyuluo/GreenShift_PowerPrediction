from Helper import DataPreprocessorTiDE
from darts.metrics import mape
from darts.models import TiDEModel
from sklearn.preprocessing import MinMaxScaler
from TiDE import build_tide_1

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
    val_size=24,
    add_time_features=True,
    scaler=sc
)

# data = data_loader.get_period_data()
# print(data)
# future = data_loader.future_series('downstream_chwsstpt', 30)
# print(future.pd_dataframe())

# Build the model
# *******************************************************************************
model = build_tide_1(
    data_loader,
    input_chunk_length=48,
    output_chunk_length=10,
    epochs=10,
    batch_size=32,
    use_future_covs=False)

# Evaluate the prediction
# *******************************************************************************
# mape = mape(test_target_series, pred)
# print('TiDE Model\'s mape is: {}'.format(mape))

# Save models
# *******************************************************************************
model.save('models/model_tide_time.pkl')
