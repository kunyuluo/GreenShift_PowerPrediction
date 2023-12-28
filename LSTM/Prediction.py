import pickle
import numpy as np
import pandas as pd
from Helper import GSDataProcessor, PredictAndForecast, Evaluate, plot_results, plot_sample_results


# Load the model
# *************************************************************************
model_index = 1
with open('models/model_lstm_{}.pkl'.format(model_index), 'rb') as f:
    model = pickle.load(f)

# Prepare the data
# *************************************************************************
file_path = '../new_data.csv'
features_name = ['cp_power', 'oat', 'oah', 'downstream_chwsstpt']
data = GSDataProcessor(
    file_path,
    feature_names=features_name,
    # start_month=10,
    # start_day=16,
    # end_month=10,
    # end_day=22,
    hour_range=(8, 20),
    group_freq=5,
    n_input=24,
    n_output=1)

train = data.train
test = data.test
# test = test.reshape(test.shape[0] * test.shape[1], test.shape[2])
# print(test)

delta = -0.5
for item_1d in train:
    for item_2d in item_1d:
        # item_2d[3] = item_2d[3] + delta
        item_2d[3] = 46.5

for item_1d in test:
    for item_2d in item_1d:
        # item_2d[3] = item_2d[3] + delta
        item_2d[3] = 46.5

# Predict for the entire test set:
# *************************************************************************
prediction = PredictAndForecast(model, train, test, n_input=24, n_output=1)
predict_values = prediction.get_predictions()
# predict_values = prediction.get_sample_prediction(4)
# predict_values = prediction.walk_forward_validation(10, 10)
# print(predict_values[0])
# print(predict_values[1])

values = predict_values.reshape(len(predict_values))
df = pd.DataFrame(values)
df.to_csv('prediction_test_cons.csv')

# Predict for one sample from the test set:
# *************************************************************************
# prediction = PredictAndForecast(model, data.train, data.test, n_input=24, n_output=3)
# predict_values, actual_values = prediction.get_sample_prediction(4)[0], prediction.get_sample_prediction(4)[1]
# print(predict_values)
# print(actual_values)

# Evaluate the prediction
# *************************************************************************
# evals = Evaluate(prediction.updated_test(), predict_values)
# print('LSTM Model\'s mape is: {}%'.format(round(evals.mape*100, 1)))
# print('LSTM Model\'s var ratio is: {}%'.format(round(evals.var_ratio*100, 1)))

# Visualize the results
# *************************************************************************
# plot_results(prediction.updated_test(), predict_values, data.df)
# plot_sample_results(actual_values, predict_values)

