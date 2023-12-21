import pickle
import numpy as np
from pandas import DataFrame
from Helper import GSDataProcessor, PredictAndForecast, Evaluate, plot_results


# Load the model
# *************************************************************************
model_index = 4
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
    n_input=6,
    n_output=6)

# print(data.X_test.shape)
# index = 4
# sample_x = data.X_test[index].reshape(1, len(data.X_test[0]), 1)
# sample_y = data.y_test[index].reshape(1, len(data.y_test[0]))
# print(data.test)

prediction = PredictAndForecast(model, data.train, data.test, n_input=6, n_output=6)
predict_values = prediction.get_predictions()

# Evaluate the prediction
# *************************************************************************
evals = Evaluate(prediction.updated_test(), predict_values)
print('Uni_LSTM Model\'s mape is: {}%'.format(round(evals.mape*100, 1)))
print('Uni_LSTM Model\'s var ratio is: {}%'.format(round(evals.var_ratio*100, 1)))

# Visualize the results
# *************************************************************************
plot_results(prediction.updated_test(), predict_values, data.df)
