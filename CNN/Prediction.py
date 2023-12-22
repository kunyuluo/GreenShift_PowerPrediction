import pickle
import numpy as np
from pandas import DataFrame
from Helper import GSDataProcessor, PredictAndForecast, Evaluate, plot_results


# Load the model
# *************************************************************************
with open('models/model_cnn.pkl', 'rb') as f:
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
    n_input=12,
    n_output=6)

prediction = PredictAndForecast(model, data.train, data.test, n_input=12, n_output=6)
predict_values = prediction.get_predictions()

# Evaluate the prediction
# *************************************************************************
evals = Evaluate(prediction.updated_test(), predict_values)
print('CNN Model\'s mape is: {}%'.format(round(evals.mape*100, 1)))
print('CNN Model\'s var ratio is: {}%'.format(round(evals.var_ratio*100, 1)))

# Visualize the results
# *************************************************************************
plot_results(prediction.updated_test(), predict_values, data.df)
