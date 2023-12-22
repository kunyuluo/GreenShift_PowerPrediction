from Helper import GSDataProcessor, plot_metrics
from CNN import build_cnn
import pickle

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

# Build the model
# *************************************************************************
epochs = 100
batch_size = 32

baseline = build_cnn(data, epochs=epochs, batch_size=batch_size)

model = baseline[0]
history = baseline[1]

# Save models
# *************************************************************************
with open('models/model_cnn.pkl', 'wb') as f:
    pickle.dump(model, f)

# Check metrics
# *************************************************************************
plot_metrics(history, epochs=epochs)
