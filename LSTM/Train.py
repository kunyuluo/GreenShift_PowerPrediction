from Helper import GSDataProcessor, PredictAndForecast, Evaluate
from Visualization import VisualizeData
from LSTM import build_lstm_1, build_lstm_2, build_lstm_3, build_lstm_4
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

# Prepare the data
# *************************************************************************
file_path = '../Data/new_data_0129.csv'
features_name = ['cp_power', 'oat', 'oah', 'downstream_chwsstpt']
n_input, n_output = 6, 6
time_feature = False

sc = MinMaxScaler(feature_range=(0, 1))
sc2 = StandardScaler()
data = GSDataProcessor(
    file_path,
    feature_names=features_name,
    test_size=0.2,
    start_date=(2023, 8, 18),
    end_date=(2023, 10, 16),
    hour_range=(7, 20),
    group_freq=5,
    n_input=n_input,
    n_output=n_output,
    add_time_features=time_feature,
    scaler=sc)

# Build the model
# *************************************************************************
model_index = 1
epochs = 30
batch_size = 24

if model_index == 1:
    baseline = build_lstm_1(data, epochs=epochs, batch_size=batch_size, lstm_dim=250, dense_dim=120)
elif model_index == 2:
    baseline = build_lstm_2(data, epochs=epochs, batch_size=batch_size)
elif model_index == 3:
    baseline = build_lstm_3(data, epochs=epochs, batch_size=batch_size)
else:
    baseline = build_lstm_4(data, epochs=epochs, batch_size=batch_size)

model = baseline[0]
history = baseline[1]

# Evaluate how good it is on the validation set, using MAPE
train = data.train
test = data.test

# Predict for the entire test set:
# *******************************************************************************
prediction = PredictAndForecast(model, train, test, n_input=n_input, n_output=n_output)
preds = prediction.get_predictions()
actuals = prediction.updated_test()
# preds = inverse_transform_prediction(prediction.get_predictions(), len(features_name), sc)
# actuals = sc.inverse_transform(prediction.updated_test())

evals = Evaluate(actuals, preds)
mape = round(evals.mape * 100, 2)
print('LSTM Model\'s mape is: {}%'.format(round(evals.mape*100, 1)))

VisualizeData.plot_results(actuals, preds, ylim=(0, 1))
# Save models
# *************************************************************************
# with open('models/model_lstm_{}.pkl'.format(model_index), 'wb') as f:
#     pickle.dump(model, f)

# Check metrics
# *************************************************************************
# VisualizeData.plot_metrics(history, epochs=epochs)
