from Helper import DataPreprocessorTiDE
from sklearn.preprocessing import MinMaxScaler
from TiDE import build_tide_1

# Get data from specific column
# *******************************************************************************
file_path = '../Data/new_data_0129.csv'
target_names = ['cp_power']
dynamic_cov_names = ['oat', 'oah', 'downstream_chwsstpt']
n_input, n_output = 47, 7

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
    val_size=0,
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
    input_chunk_length=n_input,
    output_chunk_length=n_output,
    epochs=100,
    batch_size=32,
    use_rin=True,
    use_future_covs=False)

# Save models
# *******************************************************************************
model.save('models/model_opt.pkl')
