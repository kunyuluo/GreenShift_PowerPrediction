import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
from Helper import GSDataProcessor


def build_lstm_1(dataset: GSDataProcessor, epochs=25, batch_size=32):
    """
      Builds, compiles, and fits our Uni_LSTM baseline model.
    """

    n_timesteps, n_features, n_outputs = dataset.X_train.shape[1], dataset.X_train.shape[2], dataset.y_train.shape[1]
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    opt = Adam(learning_rate=0.001)
    model = Sequential()
    model.add(LSTM(230, input_shape=(n_timesteps, n_features)))
    model.add(Dense(50))
    model.add(Dense(n_outputs))

    print("compliling baseline model")
    model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mape'])

    print("fitting model")
    history = model.fit(dataset.X_train, dataset.y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(dataset.X_test, dataset.y_test), verbose=1)

    return model, history


def build_lstm_2(dataset: GSDataProcessor, epochs=25, batch_size=32):
    """
    Builds, compiles, and fits our Uni_LSTM baseline model.
    """
    n_timesteps, n_features, n_outputs = dataset.X_train.shape[1], dataset.X_train.shape[2], dataset.y_train.shape[1]
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    # opt = Adam(learning_rate=0.0001)
    model = Sequential()
    model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features)))
    model.add(LSTM(64, activation='relu'))
    # model.add(Uni_LSTM(16))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_outputs))

    print('compiling baseline model...')
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    print('fitting model...')
    history = model.fit(dataset.X_train, dataset.y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(dataset.X_test, dataset.y_test), verbose=1, callbacks=callbacks)

    return model, history


def build_lstm_3(dataset: GSDataProcessor, epochs=25, batch_size=48):
    """
    Builds, compiles, and fits our Uni_LSTM baseline model.
    """
    n_timesteps, n_features, n_outputs = dataset.X_train.shape[1], dataset.X_train.shape[2], dataset.y_train.shape[1]
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    opt = Adam(learning_rate=0.0001)
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=64))
    model.add(Dropout(0.1))
    # model.add(Uni_LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(Uni_LSTM(units=50))
    # model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(n_outputs))

    print('compiling baseline model...')
    model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mape'])

    print('fitting model...')
    history = model.fit(dataset.X_train, dataset.y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(dataset.X_test, dataset.y_test), verbose=1, callbacks=callbacks)

    return model, history
