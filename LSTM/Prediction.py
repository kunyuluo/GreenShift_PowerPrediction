import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from Helper import GSDataProcessor, PredictAndForecast, Evaluate
from Helper import inverse_transform_prediction, scale_data
from Visualization import VisualizeData


# Load the model
# *************************************************************************
model_index = 1
with open('models/model_lstm_{}.pkl'.format(model_index), 'rb') as f:
    model = pickle.load(f)

# Load the scaler
# *************************************************************************
# with open('scalers/scaler.pkl', 'rb') as s:
#     scaler = pickle.load(f)

# Prepare the data
# *************************************************************************
file_path = '../Data/new_data_0102.csv'
features_name = ['cp_power', 'oat', 'oah', 'downstream_chwsstpt']
n_input, n_output = 24, 1

sc = MinMaxScaler(feature_range=(0, 1))
data = GSDataProcessor(
    file_path,
    feature_names=features_name,
    test_size=0.2,
    hour_range=(6, 20),
    group_freq=5,
    n_input=n_input,
    n_output=n_output,
    scaler=sc)

train = data.train
test = data.test
# test = test.reshape(test.shape[0] * test.shape[1], test.shape[2])
# print(test)

# delta = -0.5
# constant = scale_data(46, (41, 49))
# for item_1d in train:
#     for item_2d in item_1d:
#         # item_2d[3] = item_2d[3] + delta
#         item_2d[3] = constant
#
# for item_1d in test:
#     for item_2d in item_1d:
#         # item_2d[3] = item_2d[3] + delta
#         item_2d[3] = constant

# Predict for the entire test set:
# *************************************************************************
prediction = PredictAndForecast(model, train, test, n_input=n_input, n_output=n_output)
# predict_values = prediction.get_predictions()
# actual_values = prediction.updated_test()
predict_values = inverse_transform_prediction(prediction.get_predictions(), len(features_name), sc)
actual_values = sc.inverse_transform(prediction.updated_test())

# Walk-Forward Predict for certain length of step:
# *************************************************************************
# walk_forward = prediction.walk_forward_validation(8, 250)
# predict_values = inverse_transform_prediction(walk_forward[0], len(features_name), sc)
# actual_values = inverse_transform_prediction(walk_forward[1], len(features_name), sc)

# values = predict_values.reshape(len(predict_values))
# df = pd.DataFrame(values)
# df.to_csv('trend_test_0104_46.csv')

# Predict for one sample from the test set:
# *************************************************************************
# sample_index = 521
# # prediction = PredictAndForecast(model, data.train, data.test, n_input=n_input, n_output=n_output)
# predict_values = inverse_transform_prediction(prediction.get_sample_prediction(sample_index)[0], len(features_name), sc)
# actual_values = inverse_transform_prediction(prediction.get_sample_prediction(sample_index)[1], len(features_name), sc)
# print(predict_values)
# print(actual_values)

# Evaluate the prediction
# *************************************************************************
evals = Evaluate(actual_values, predict_values)
print('LSTM Model\'s mape is: {}%'.format(round(evals.mape*100, 1)))
print('LSTM Model\'s var ratio is: {}%'.format(round(evals.var_ratio*100, 1)))

# Visualize the results
# *************************************************************************
# VisualizeData.plot_results(actual_values, predict_values)
VisualizeData.plot_results(actual_values, predict_values, ylim=(0, 180))
# VisualizeData.plot_sample_results(actual_values, predict_values)

