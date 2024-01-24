import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class VisualizeData:
    @staticmethod
    def check_data_distribution(df: pd.DataFrame, column_name: str = 'cp_power', log_transform: bool = False):
        """
        Check the distribution of the data.
        """
        data = df[column_name]

        if log_transform:
            data = np.log(data)

        plt.hist(data, bins=20)
        plt.title('Distribution of \'{}\''.format(column_name))
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.show()

    @staticmethod
    def check_linearity(
            df: pd.DataFrame,
            column_name_1: str,
            column_name_2: str,
            switch_table: bool = False,
            log_transform: bool = False):
        """
        Check the linearity of the data.
        """
        data_1 = df[column_name_1]
        data_2 = df[column_name_2]

        if log_transform:
            data_1 = np.log(data_1)
            # data_2 = np.log(data_2)

        x_label = column_name_1
        y_label = column_name_2

        if switch_table:
            data_1, data_2 = data_2, data_1
            x_label, y_label = y_label, x_label

        plt.scatter(data_1, data_2, s=2)
        plt.title('Linearity between \'{}\' and \'{}\''.format(column_name_1, column_name_2))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    @staticmethod
    def check_autocorrelation(df: pd.DataFrame, column_name: str = 'cp_power'):
        """
        Check the autocorrelation of the data.
        """
        data = df[column_name]

        pd.plotting.lag_plot(data)
        plt.show()

    @staticmethod
    def plot_variable(df: pd.DataFrame, var_name: str = 'cp_power', is_daily: bool = True):
        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot()

        if is_daily:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        # ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax.plot(df.index, df[var_name], color='black')
        ax.set_ylim(0, )

        plt.show()

    @staticmethod
    def plot_variable_no_time(df: pd.DataFrame, var_name: str = 'cp_power', y_limit: tuple = None):
        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot()

        x = range(len(df[var_name]))
        ax.plot(x, df[var_name], color='black')
        if y_limit is not None:
            ax.set_ylim(y_limit)
        else:
            ax.set_ylim(0, )

        plt.show()

    @staticmethod
    def plot_metrics(history, epochs: int = 25):
        acc = history.history['mape']
        val_acc = history.history['val_mape']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training MAPE')
        plt.plot(epochs_range, val_acc, label='Validation MAPE')
        plt.legend(loc='upper right')
        # plt.ylim(0, 50)
        plt.title('Training and Validation MAPE')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        # plt.ylim(0, 100)
        plt.title('Training and Validation Loss')
        plt.show()

    @staticmethod
    def plot_results(test, preds, title_suffix=None, xlabel='Power Prediction', ylim=None):
        """
        Plots training data in blue, actual values in red, and predictions in green, over time.
        """
        fig, ax = plt.subplots(figsize=(18, 6))
        # x = df.Close[-498:].index
        if test.shape[1] > 1:
            test = test[:, 0]

        plot_test = test[0:]
        plot_preds = preds[0:]

        # x = df[-(plot_test.shape[0] * plot_test.shape[1]):].index
        # plot_test = plot_test.reshape((plot_test.shape[0] * plot_test.shape[1], 1))
        plot_preds = plot_preds.reshape((plot_preds.shape[0] * plot_preds.shape[1], 1))

        ax.plot(plot_test, label='actual')
        ax.plot(plot_preds, label='preds')

        if title_suffix is None:
            ax.set_title('Predictions vs. Actual')
        else:
            ax.set_title(f'Predictions vs. Actual, {title_suffix}')

        ax.set_xlabel('Date')
        ax.set_ylabel(xlabel)
        ax.legend()
        if ylim is not None:
            ax.set_ylim(ylim)

        plt.show()

    @staticmethod
    def plot_sample_results(test, preds):
        plt.plot(test, color='black', label='Actual')
        plt.plot(preds, color='green', label='Predicted')
        # plt.title('Power Prediction')
        # plt.xlabel('Hour')
        # plt.ylabel('kWh')
        plt.legend()
        plt.ylim(0, 200)
        plt.show()
