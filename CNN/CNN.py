import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import Adam
from Helper import GSDataProcessor


def build_cnn(dataset: GSDataProcessor, epochs=25, batch_size=32):
    """
    Builds, compiles, and fits our CNN baseline model.
    """

    n_timesteps, n_features, n_outputs = dataset.X_train.shape[1], dataset.X_train.shape[2], dataset.y_train.shape[1]
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    opt = Adam(learning_rate=0.0001)
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_outputs))

    print("compliling baseline model")
    model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mape'])

    print("fitting model")
    history = model.fit(dataset.X_train, dataset.y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(dataset.X_test, dataset.y_test), verbose=1)

    return model, history
