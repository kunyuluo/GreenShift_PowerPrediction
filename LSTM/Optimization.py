import optuna
from Helper import GSDataProcessor
from Helper import PredictAndForecast, Evaluate
from Helper import inverse_transform_prediction
from LSTM import build_lstm_1
from sklearn.preprocessing import MinMaxScaler

# Prepare the data
# *************************************************************************
file_path = '../Data/new_data_0129.csv'
features_name = ['cp_power', 'oat', 'oah', 'downstream_chwsstpt']
sc = MinMaxScaler(feature_range=(0, 1))
n_output = 6


# Define objectuve function.
def objective(trial):
    # select input and output chunk lengths
    in_len = trial.suggest_int("in_len", 6, 30, step=6)
    # out_len = trial.suggest_int("out_len", 8, 10)
    lstm_dim = trial.suggest_int("lstm_dim", 150, 350, step=10)
    dense_dim = trial.suggest_int("dense_dim", 30, 150, step=10)
    batch_size = trial.suggest_int("batch_size", 16, 64, step=8)
    use_time_covs = trial.suggest_categorical("use_time_covs", [True, False])

    data_loader = GSDataProcessor(
        file_path,
        feature_names=features_name,
        test_size=0.2,
        start_date=(2023, 8, 18),
        end_date=(2023, 10, 16),
        hour_range=(6, 20),
        group_freq=5,
        n_input=in_len,
        n_output=n_output,
        scaler=sc,
        add_time_features=use_time_covs)

    model = build_lstm_1(
        data_loader,
        epochs=30,
        batch_size=batch_size,
        lstm_dim=lstm_dim,
        dense_dim=dense_dim)

    # Evaluate how good it is on the validation set, using MAPE
    train = data_loader.train
    test = data_loader.test

    # Predict for the entire test set:
    # *******************************************************************************
    prediction = PredictAndForecast(model[0], train, test, n_input=in_len, n_output=n_output)
    preds = prediction.get_predictions()
    actuals = prediction.updated_test()
    # preds = inverse_transform_prediction(prediction.get_predictions(), len(features_name), sc)
    # actuals = sc.inverse_transform(prediction.updated_test())

    evals = Evaluate(actuals, preds)
    mape = round(evals.mape * 100, 2)

    return mape if mape is not None else float("inf")


def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


# optimize hyperparameters by minimizing the MAPE on the validation set
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, callbacks=[print_callback])
