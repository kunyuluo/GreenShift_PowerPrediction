from Helper import GSDataProcessor, plot_metrics
from LSTM import build_lstm_1, build_lstm_2, build_lstm_3, build_lstm_4
from sklearn.preprocessing import MinMaxScaler
import pickle

# Prepare the data
# *************************************************************************
file_path = '../new_data.csv'
features_name = ['cp_power', 'oat', 'oah', 'downstream_chwsstpt']
n_input, n_output = 24, 6
data = GSDataProcessor(
    file_path,
    feature_names=features_name,
    test_size=0.2,
    start_month=8,
    start_day=18,
    end_month=11,
    end_day=11,
    hour_range=(8, 20),
    group_freq=5,
    n_input=n_input,
    n_output=n_output)

# Build the model
# *************************************************************************
model_index = 2
epochs = 70
batch_size = 32

if model_index == 1:
    baseline = build_lstm_1(data, epochs=epochs, batch_size=batch_size)
elif model_index == 2:
    baseline = build_lstm_2(data, epochs=epochs, batch_size=batch_size)
elif model_index == 3:
    baseline = build_lstm_3(data, epochs=epochs, batch_size=batch_size)
else:
    baseline = build_lstm_4(data, epochs=epochs, batch_size=batch_size)

model = baseline[0]
history = baseline[1]

# Save models
# *************************************************************************
with open('models/model_lstm_{}.pkl'.format(model_index), 'wb') as f:
    pickle.dump(model, f)

# Check metrics
# *************************************************************************
plot_metrics(history, epochs=epochs)
