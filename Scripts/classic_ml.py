import pandas as pd
import numpy as np
from time import time

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.base import clone


def get_elapsed_time(model, x, num_samples=200):
    """
    Returns time elapsed during evaluation.

    Parameters
    ----------
    model : model object
        Trained model evaluation time to be measured for.
    x : pd.DataFrame
        Dataframe that contains data to input into the model.
    num_samples : int
        Number of samples to be used for evaluation (default 200).

    Returns
    ----------
    elapsed_time : float
        Elapsed time.
    """
    samples_x = x.iloc[:num_samples, :]
    time_start = time()
    samples_y = model.predict(samples_x)
    time_stop = time()

    elapsed_time = time_stop - time_start

    return elapsed_time


def calculate_metrics(y_true, y_pred, param_names=None):
    """
    Calculates MSE and R2 metrics for all parameters in dataset.

    Parameters
    ----------
    y_true : pd.DataFrame
        True values of Y-data.
    y_pred : pd.DataFrame
        Predicted values of Y-data.
    param_names : list of str or None
        Parameter names (format `M{mode} Param_name`) to be used for metrics calculations. If None, uses all params in y_true (default None).

    Returns
    ----------
    output_dict : dict
        Dictionary that contains param name as a key and list of 2 values (MSE and R2 metrics) for that param. For instance, {'M1 Eigenfrequency (Hz)': [0.01, 0.95]}
    """
    output_dict = {}
    
    if param_names is None:
        param_names = list(y_true.columns)
    
    y_true = y_true.values

    for index, name in enumerate(param_names):
        output_dict[name] = []
        output_dict[name].append(mean_squared_error(y_true[:, index], y_pred[:, index]))
        output_dict[name].append(r2_score(y_true[:, index], y_pred[:, index]))

    return output_dict


def plot_metrics(output_dict, apply_log_mse, apply_log_r2, font_size=1, param_names=None):
    """
    Plots metrics from dictionary. Format of plotting: (2 x number of parameters (not considering different modes) plots.
    Rows: MSE and R2 metrics, columns: parameters (e.g. 'Eigenfrequency (Hz)', 'Quality factor', ...).
    Vertical axis: metric value, horizontal axis: mode number

    Parameters
    ----------
    output_dict : dict
        Dictionary that contains param name as a key and list of 2 values (MSE and R2 metrics) for that param. For instance, {'M1 Eigenfrequency (Hz)': [0.01, 0.95]}
    apply_log_mse : bool
        If True, y-axis of all MSE metric plots will be log-scaled.
    apply_log_r2 : bool
        If True, y-axis of all R2 metric plots will be log-scaled.
    font_size : float
        Coefficient for resizing font (default 1).
    param_names : list of str or None
        Parameter names (format `Param_name` without mode number specification) to be used for metrics calculations. If None, uses all params in output_dict.keys() (defaul None).
    """
    if param_names is None:
        param_names = ['Eigenfrequency (Hz)', 'Quality factor', 'Effective mass (kg)', 'TED (W)', 'Noise (kg^2/s^3)']

    fig, ax = plt.subplots(nrows=2,
                           ncols=len(param_names),
                           figsize=(5 * len(param_names), 5 * 2),
                           sharey='row')
    modes = [1, 2, 3, 4]

    # MSE metric visualization
    for index, name in enumerate(param_names):
        points_to_plot = []

        for mode_number in modes:
            points_to_plot.append(output_dict[f'M{mode_number} {name}'][0])

        ax[0, index].plot(modes, points_to_plot)
        ax[0, index].set_title(f"{name} MSE Loss.", fontsize=2 * font_size * (len(param_names) + 1))
        ax[0, index].set_xlabel('Mode number')
        ax[0, index].set_ylabel('MSE Loss')
        ax[0, index].set_xticks(modes)
        ax[0, index].grid(visible=True)

        if apply_log_mse:
            ax[0, index].set_yscale('log')

    # R2 metric visualization
    for index, name in enumerate(param_names):
        points_to_plot = []

        for mode_number in modes:
            points_to_plot.append(output_dict[f'M{mode_number} {name}'][1])

        ax[1, index].plot(modes, points_to_plot)
        ax[1, index].set_title(f"{name} R2 Score.", fontsize=2 * font_size * (len(param_names) + 1))
        ax[1, index].set_xlabel('Mode number')
        ax[1, index].set_ylabel('R2 Score')
        ax[1, index].set_xticks(modes)
        ax[1, index].grid(visible=True)

        if apply_log_r2:
            ax[1, index].set_yscale('log')

    plt.show()


def plot_metrics_dense(output_dict, apply_log_mse, apply_log_r2, font_size=1, param_names=None):
    """
    Plots metrics from dictionary. Format of plotting: (1 x 2) plots.
    Columns: MSE and R2 metrics.
    Vertical axis: metric value, horizontal axis: mode number, color legend: different parameters.

    Parameters
    ----------
    output_dict : dict
        Dictionary that contains param name as a key and list of 2 values (MSE and R2 metrics) for that param. For instance, {'M1 Eigenfrequency (Hz)': [0.01, 0.95]}
    apply_log_mse : bool
        If True, y-axis of all MSE metric plots will be log-scaled.
    apply_log_r2 : bool
        If True, y-axis of all R2 metric plots will be log-scaled.
    font_size : float
        Coefficient for resizing font (default 1).
    param_names : list of str or None
        Parameter names (format `Param_name` without mode number specification) to be used for metrics calculations. If None, uses all params in output_dict.keys() (defaul None).
    """
    if param_names is None:
        param_names = ['Eigenfrequency (Hz)', 'Quality factor', 'Effective mass (kg)', 'TED (W)', 'Noise (kg^2/s^3)']

    fig, ax = plt.subplots(nrows=1,
                           ncols=2,
                           figsize=(5 * 2, 5))
    modes = [1, 2, 3, 4]

    # MSE metric visualization
    for name in param_names:
        points_to_plot = []

        for mode_number in modes:
            points_to_plot.append(output_dict[f'M{mode_number} {name}'][0])

        ax[0].plot(modes, points_to_plot, label=f'{name}')

    if apply_log_mse:
        ax[0].set_yscale('log')

    ax[0].set_title("MSE Loss.", fontsize=2 * 3 * font_size)
    ax[0].set_xlabel('Mode number')
    ax[0].set_ylabel('MSE Loss')
    ax[0].set_xticks(modes)
    ax[0].grid(visible=True)
    ax[0].legend()

    # R2 metric visualization
    for name in param_names:
        points_to_plot = []

        for mode_number in modes:
            points_to_plot.append(output_dict[f'M{mode_number} {name}'][1])

        ax[1].plot(modes, points_to_plot, label=f'{name}')

    if apply_log_r2:
        ax[1].set_yscale('log')

    ax[1].set_title("R2 Score.", fontsize=2 * 3 * font_size)
    ax[1].set_xlabel('Mode number')
    ax[1].set_ylabel('R2 Score')
    ax[1].set_xticks(modes)
    ax[1].grid(visible=True)
    ax[1].legend()

    plt.show()


def compare_models(dict_list, model_names, apply_log_mse, apply_log_r2, font_size=1, sharey='row', modes=None, param_names=None):
    """
    Plots metrics of specified models on the same plot. Format of plotting: (2 x 5) plots.
    Rows: MSE and R2 metrics, columns: parameters (e.g. 'Eigenfrequency (Hz)', 'Quality factor', ... )
    Vertical axis: metric value, horizontal axis: mode number, color legend: different models.

    Parameters
    ----------
    dict_list : list of dict
        List that contains dictionaries with metrics. For instance, [{'M1 Eigenfrequency (Hz)': [0.01, 0.95], ...}, ... ].
    model_names : list of str
        List that contains names of models. Order of these name should correspond to the order of dictionaries in dict_list.
    apply_log_mse : bool
        If True, y-axis of all MSE metric plots will be log-scaled.
    apply_log_r2 : bool
        If True, y-axis of all R2 metric plots will be log-scaled.
    font_size : float
        Coefficient for resizing font (default 1).
    sharey : bool or {'none', 'all', 'row', 'col'} (default 'row').
        Controls sharing of properties among x (*sharex*) or y (*sharey*) axes:
        - True or 'all': x- or y-axis will be shared among all subplots.
        - False or 'none': each subplot x- or y-axis will be independent.
        - 'row': each subplot row will share an x- or y-axis.
        - 'col': each subplot column will share an x- or y-axis.
    modes : list of int
        List that contains mode numbers (default [1, 2, 3, 4]).
    param_names : list of str or None
        Parameter names (`M{mode} Param_name` without mode number specification) to be used for metrics calculations. If None, uses all params in output_dict.keys() (defaul None).
    """
    if modes is None:
        modes = [1, 2, 3, 4]

    if param_names is None:
        param_names = ['Eigenfrequency (Hz)', 'Quality factor', 'Effective mass (kg)', 'TED (W)', 'Noise (kg^2/s^3)']

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(25, 10), sharey=sharey)
    
    # MSE and R2 rows
    for row_index in range(0, 2):
        
        # Columns for each parameter
        for col_index in range(0, len(param_names)):
            
            # Color-labeled model metrics
            for model_index in range(0, len(model_names)):
                points_to_plot = []
                for mode in modes:
                    points_to_plot.append(dict_list[model_index][f'M{mode} {param_names[col_index]}'][row_index])

                ax[row_index, col_index].plot(modes, points_to_plot, label=model_names[model_index])

            # MSE plot post-processing
            if row_index == 0:
                if apply_log_mse:
                    ax[row_index, col_index].set_yscale('log')

                ax[row_index, col_index].set_ylabel('MSE Loss')
                ax[row_index, col_index].set_title(f"{param_names[col_index]} MSE Loss", fontsize=2 * 6 * font_size)

            # R2 plot post-processing
            if row_index == 1:
                if apply_log_r2:
                    ax[row_index, col_index].set_yscale('log')

                ax[row_index, col_index].set_ylabel('R2 Loss')
                ax[row_index, col_index].set_title(f"{param_names[col_index]} R2 Score", fontsize=2 * 6 * font_size)

            ax[row_index, col_index].set_xlabel('Mode number')
            ax[row_index, col_index].tick_params(axis='y', labelleft=True)
            ax[row_index, col_index].set_xticks(modes)
            ax[row_index, col_index].grid(visible=True)
            ax[row_index, col_index].legend()
    plt.plot()

    
def plot_distribution(df, param_name, log_scale, font_size=1, modes=None):
    """
    Plots distribution of specified parameter for resonant modes.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that contains specified parameter.
    param_name : str
        Specified parameter name. Note: no need to include mode number in param_name!
    log_scale : bool
        If True, y-axis of all plots will be log-scaled.
    font_size : float
        Coefficient for resizing font (default 1).
    modes : list of int
        List that contains mode numbers (default [1, 2, 3, 4]).
    """
    if modes is None:
        modes = [1, 2, 3, 4]

    col_names_to_plot = []
    
    for mode in modes:
        col_names_to_plot.append(f'M{mode} {param_name}')
    
    fig, ax = plt.subplots(nrows=1, ncols=len(col_names_to_plot), figsize=(5 * len(col_names_to_plot), 5))
    
    if len(col_names_to_plot) == 1:
        ax = [ax]
    
    for j in range(0, len(col_names_to_plot)):
        sns.histplot(df.loc[:, col_names_to_plot[j]], ax=ax[j], log_scale=log_scale)
        ax[j].set_title(f"{col_names_to_plot[j]} distribution.", fontsize=2 * font_size * (len(col_names_to_plot[j]) + 1))
        ax[j].set_xlabel(col_names_to_plot[j])
        ax[j].set_ylabel('Count')
    plt.show()


class CustomCV:
    def __init__(self, model, n_splits, shuffle, random_state=42):
        """
        Class that performs custom cross-validation.

        Parameters
        ----------
        model : model object
            Non-trained model to be set as basic model for cross-validation.
        n_splits : int
            Number of splits in cross-validation.
        shuffle : bool
            If True, shuffles data during cross-validation.
        random_state : int
            Random state for reproducibility (default 42). `random_state` is passed into internal cross-validation function.
        """
        self.kf_sklearn = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.model = model
        self.n_splits = n_splits

        # Index logs
        self.train_indices_log = []
        self.val_indices_log = []

        # Metrics log
        self.metrics_log = []

        # Predictions log
        self.y_pred_train_log = []
        self.y_pred_val_log = []

        self.x_trainval = None
        self.y_trainval = None

        self.param_names_x = None
        self.param_names_y = None

    # Функция, осуществляющая кастомную кросс-валидацию
    def cross_validate(self, x_trainval, y_trainval, return_indices):
        """
        Initializes the process of cross-validation.

        Parameters
        ----------
        x_trainval : pd.DataFrame
            Training and validation X-data.
        y_trainval : pd.DataFrame
            Training and validation Y-data.
        return_indices : bool
            If True, returns

        Returns
        ----------
        metrics_log : list of dict
            List that contains dictionaries with metrics values. Each dictionary index in metrics_log corresponds to the index of particular fold. E.g. index = 2 means this dictionary describes metrics values on the fold with index = 2.
        train_indices_log : list[list] of int
            List that contains train object index logs for all folds. Is returned only if return_indices is True.
        val_indices_log : list[list] of int
            List that contains validation object index logs for all folds. Is returned only if return_indices is True.
        """
        # Copy data
        self.x_trainval = x_trainval
        self.y_trainval = y_trainval

        # Copy column names
        self.param_names_x = list(x_trainval.columns)
        self.param_names_y = list(y_trainval.columns)

        output_dict = {}
        counter = 1
        param_names = list(y_trainval.columns)

        # Iterating over folds
        for train_indices, val_indices in tqdm(self.kf_sklearn.split(x_trainval.values, y_trainval.values)):
            # Casting indices to `int` type
            train_indices = train_indices.astype('int')
            val_indices = val_indices.astype('int')

            # Save indices to logs
            self.train_indices_log.append(train_indices)
            self.val_indices_log.append(val_indices)

            # Creating a fresh copy of the base model for this fold and training it
            model = clone(self.model)
            model.fit(x_trainval.iloc[train_indices, :].values, y_trainval.iloc[train_indices, :].values)

            # Making predictions on this fold
            y_pred_train = model.predict(x_trainval.iloc[train_indices, :].values)
            y_pred_val = model.predict(x_trainval.iloc[val_indices, :].values)

            # Save metrics to log
            self.metrics_log.append(
                calculate_metrics(y_true=y_trainval.iloc[val_indices, :], y_pred=y_pred_val, param_names=param_names))

            print(f' Fold N{counter} is processed!')
            counter += 1

            # Save predictions to logs
            self.y_pred_train_log.append(y_pred_train)
            self.y_pred_val_log.append(y_pred_val)

        if return_indices:
            return self.metrics_log, self.train_indices_log, self.val_indices_log
        else:
            return self.metrics_log

    def plot_metrics_dense(self, fold_index, apply_log_mse, apply_log_r2, param_names=None):
        """
        Returns two ax objects with plotted metrics.

        Parameters
        ----------
        fold_index : int
            Index of a fold (within [0, n_splits - 1]) for which metrics will be plotted.
        apply_log_mse : bool
            If True, y-axis of all MSE metric plots will be log-scaled.
        apply_log_r2 : bool
            If True, y-axis of all R2 metric plots will be log-scaled.
        param_names : list of str or None
            Parameter names (`M{mode} Param_name` without mode number specification) to be used for metrics calculations. If None, uses all params in output_dict.keys() (defaul None).

        Returns
        ----------
        ax_mse : axes object
            Axes object with values for MSE metric plotted.
        ax_r2 : axes object
            Axes object with values for R2 metric plotted.
        """
        if param_names is None:
            param_names = ['Eigenfrequency (Hz)', 'Quality factor', 'Effective mass (kg)', 'TED (W)',
                           'Noise (kg^2/s^3)']

        fig, ax = plt.subplots(nrows=1,
                               ncols=2,
                               figsize=(5 * 2, 5))
        modes = [1, 2, 3, 4]

        # MSE metrics visualization
        for name in param_names:
            points_to_plot = []

            for mode in modes:
                points_to_plot.append(self.metrics_log[fold_index][f'M{mode} ' + f'{name}'][0])

            ax[0].plot(modes, points_to_plot, label=f'{name}')

        if apply_log_mse:
            ax[0].set_yscale('log')

        ax[0].set_title("MSE Loss.", fontsize=2 * 3)
        ax[0].set_xlabel('Mode number')
        ax[0].set_ylabel('MSE Loss')
        ax[0].set_xticks(modes)
        ax[0].grid(visible=True)
        ax[0].legend()

        # R2 metrics visualization
        for name in param_names:
            points_to_plot = []

            for mode in modes:
                points_to_plot.append(self.metrics_log[fold_index][f'M{mode} ' + f'{name}'][1])

            ax[1].plot(modes, points_to_plot, label=f'{name}')

        if apply_log_r2:
            ax[1].set_yscale('log')

        ax[1].set_title("R2 Score.", fontsize=2 * 3)
        ax[1].set_xlabel('Mode number')
        ax[1].set_ylabel('R2 Score')
        ax[1].set_xticks(modes)
        ax[1].grid(visible=True)
        ax[1].legend()

        ax_mse, ax_r2 = ax[0], ax[1]

        return ax_mse, ax_r2

    def plot_distribution_trainval(self, fold_index, log_scale):
        """
        Plots training and validation distribution of true and predicted values.

        Parameters
        ----------
        fold_index : int
            Index of a fold (within [0, n_splits - 1]) for which metrics will be plotted.
        log_scale : bool
            If True, all plots will be log-scaled.
        """
        x_col_len, y_col_len = len(self.param_names_x), len(self.param_names_y)

        # Metrics plotting for the specified fold
        print('Started plotting metrics.')
        fig_metrics, ax_metrics = plt.subplots(nrows=1, ncols=2, figsize=(5 * 2, 5))
        ax_metrics[0], ax_metrics[1] = self.plot_metrics_dense(fold_index=fold_index,
                                                               apply_log_mse=True,
                                                               apply_log_r2=False)
        print('Metrics have been successfully plotted!')

        # ======
        # Iterating over all parameters in X
        print('Started plotting X-data distribution.')
        fig_x, ax_x = plt.subplots(nrows=1, ncols=x_col_len, figsize=(10 * x_col_len, 15))
        for index in tqdm(range(0, x_col_len)):
            # Distribution on train part of the fold in X-data
            sns.histplot(self.x_trainval.iloc[self.train_indices_log[fold_index], index],
                         ax=ax_x[index],
                         element="step",
                         log_scale=log_scale)

            # Distribution on val part of the fold in X-data
            sns.histplot(self.x_trainval.iloc[self.val_indices_log[fold_index], index],
                         ax=ax_x[index],
                         element="step",
                         log_scale=log_scale)

            ax_x[index].set_title(f"{self.param_names_x[index]} distribution.",
                                  fontsize=20)
            ax_x[index].set_xlabel(self.param_names_x[index])
            ax_x[index].set_ylabel('Count')
        print('X-data distribution has been successfully plotted!')

        # ======
        # Iterating over all parameters in Y
        print('Started plotting Y-data distribution.')
        fig_y, ax_y = plt.subplots(nrows=4, ncols=int(y_col_len / 4), figsize=(50, 40))
        for index in tqdm(range(0, y_col_len)):
            if 'M1' in self.param_names_y[index]:
                mode = 0
            elif 'M2' in self.param_names_y[index]:
                mode = 1
            elif 'M3' in self.param_names_y[index]:
                mode = 2
            elif 'M4' in self.param_names_y[index]:
                mode = 3

            # Distribution on train part of the fold in X-data (true values)
            sns.histplot(self.y_trainval.iloc[self.train_indices_log[fold_index], index],
                         ax=ax_y[mode, index - 5 * mode],
                         element="step",
                         log_scale=log_scale)

            # Distribution on val part of the fold in X-data (true values)
            sns.histplot(self.y_trainval.iloc[self.val_indices_log[fold_index], index],
                         ax=ax_y[mode, index - 5 * mode],
                         element="step",
                         log_scale=log_scale)

            # Distribution on train part of the fold in X-data (predicted values)
            sns.histplot(self.y_pred_train_log[fold_index][:, index],
                         ax=ax_y[mode, index - 5 * mode],
                         element="step",
                         fill=False,
                         color='k',
                         log_scale=log_scale)

            # Distribution on val part of the fold in X-data (predicted values)
            sns.histplot(self.y_pred_val_log[fold_index][:, index],
                         ax=ax_y[mode, index - 5 * mode],
                         element="step",
                         fill=False,
                         color='k',
                         log_scale=log_scale)

            ax_y[mode, index - 5 * mode].set_title(
                f"{self.param_names_y[index]} distribution. MSE = {self.metrics_log[fold_index][self.param_names_y[index]][0]:.3f}, R2 = {self.metrics_log[fold_index][self.param_names_y[index]][1]:.3f}",
                fontsize=12)
            ax_y[mode, index - 5 * mode].set_xlabel(self.param_names_y[index])
            ax_y[mode, index - 5 * mode].set_ylabel('Count')
        print('Y-data disribution has been successfully plotted!')
        plt.plot()