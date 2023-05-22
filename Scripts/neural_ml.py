import pandas as pd
import numpy as np
from time import time

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from torch.nn.functional import mse_loss
from torcheval.metrics.functional import r2_score

from IPython.display import clear_output
from collections import defaultdict


def calculate_metrics_torch(y_true, y_pred, param_names=None):
    """
    Calculates MSE and R2 metrics for all parameters in dataset.

    Parameters
    ----------
    y_true : torch.tensor
        True values of Y-data.
    y_pred : torch.tensor
        Predicted values of Y-data.
    param_names : list of str or None
        Parameter names (format `M{mode} Param_name`) to be used for metrics calculations. If None, uses 5 convenient params and 4 modes (default None).

    Returns
    ----------
    output_dict : dict
        Dictionary that contains param name as a key and list of 2 values (MSE and R2 metrics) for that param. For instance, {'M1 Eigenfrequency (Hz)': [0.01, 0.95]}
    """
    if param_names is None:
        param_names = ['M1 Eigenfrequency (Hz)', 'M1 Quality factor', 'M1 Effective mass (kg)', 'M1 TED (W)', 'M1 Noise (kg^2/s^3)',
                       'M2 Eigenfrequency (Hz)', 'M2 Quality factor', 'M2 Effective mass (kg)', 'M2 TED (W)', 'M2 Noise (kg^2/s^3)',
                       'M3 Eigenfrequency (Hz)', 'M3 Quality factor', 'M3 Effective mass (kg)', 'M3 TED (W)', 'M3 Noise (kg^2/s^3)',
                       'M4 Eigenfrequency (Hz)', 'M4 Quality factor', 'M4 Effective mass (kg)', 'M4 TED (W)', 'M4 Noise (kg^2/s^3)']
    output_dict = {}

    for index, name in enumerate(param_names):
        output_dict[name] = []
        output_dict[name].append(mse_loss(input=y_pred[:, index], target=y_true[:, index]))
        output_dict[name].append(r2_score(input=y_pred[:, index], target=y_true[:, index]))

    return output_dict


class ProgressPlotter:
    def __init__(self) -> None:
        """
        Creates object instance and initializes history dictionary.
        """
        self._history_dict = defaultdict(list)

    def add_scalar(self, tag: str, value) -> None:
        """
        Adds scalar values to history dict.

        Parameters
        ----------
        tag : str
            Tag (name) of value.
        value : float
            Value to be added to history dict.
        """
        self._history_dict[tag].append(value)  # тут можно хранить разные величины

    def display_keys(self, ax, tags):
        """
        Plots values under a group of specified tags in axis.

        Parameters
        ----------
        ax : axes object
            Axes object where values for the specified tag will be plotted.
        tags : list of str
            Tags which will be plotted in the same plot.

        Returns
        ----------
        output_dict : dict
            Dictionary that contains param name as a key and list of 2 values (MSE and R2 metrics) for that param. For instance, {'M1 Eigenfrequency (Hz)': [0.01, 0.95]}
        """
        if isinstance(tags, str):
            tags = [tags]
        history_len = 0
        ax.grid()
        for key in tags:
            ax.plot(self._history_dict[key], marker="X", label=key)
            history_len = max(history_len, len(self.history_dict[key]))
        if len(tags) > 1:
            ax.legend(loc="lower left")
        else:
            ax.set_ylabel(key)
        ax.set_xlabel('step')
        ax.set_xticks(np.arange(history_len))
        ax.set_xticklabels(np.arange(history_len))

    def display(self, groups=None):
        """
        Displays all collected values for specified groups of tags.

        Parameters
        ----------
        groups : list containing lists of str
            All tags specified in 2nd level list will be plotted on the same plot. For instance, groups = [['loss_val', 'loss_train'], ['acc_val', 'acc_train']].
        """
        clear_output()
        n_groups = len(groups)
        fig, ax = plt.subplots(n_groups, 1, figsize=(12, 3 * n_groups))
        if n_groups == 1:
            ax = [ax]
        for i, keys in enumerate(groups):
            self.display_keys(ax[i], keys)
        fig.tight_layout()
        plt.show()

    @property
    def history_dict(self):
        return dict(self._history_dict)


@torch.inference_mode()
def calculate_val_metrics_mlp(model, data_loader, param_names=None):
    """
    Calculates MSE and R2 metrics on a validation dataset for fully connected network.

    Parameters
    ----------
    model : model object
        Model that is being trained.
    data_loader : torch.utils.data.DataLoader
        Validation dataset dataloader.
    param_names : list of str or None
        Parameter names (format `M{mode} Param_name`) to be used for metrics calculations. If None, uses 5 convenient params and 4 modes (default None).

    Returns
    ----------
    output_dict : dict
        Dictionary that contains param name as a key and list of 2 values (MSE and R2 metrics) for that param.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if param_names is not None:
        num_pars_y = len(param_names)
    else:
        num_pars_y = 20

    y_log = torch.empty(size=[0, num_pars_y]).to(device)
    output_log = torch.empty(size=[0, num_pars_y]).to(device)

    for batch in data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        output = model(x)
        y_log = torch.cat((y_log, y), dim=0)
        output_log = torch.cat((output_log, output), dim=0)

    output_dict = calculate_metrics_torch(y_true=y_log, y_pred=output_log, param_names=param_names)

    return output_dict


@torch.inference_mode()
def calculate_val_metrics_branched(model, data_loader, param_names=None):
    """
    Calculates MSE and R2 metrics on a validation dataset for branched separate network.

    Parameters
    ----------
    model : model object
        Model that is being trained.
    data_loader : torch.utils.data.DataLoader
        Validation dataset dataloader.
    param_names : list of str or None
        Parameter names (format `M{mode} Param_name`) to be used for metrics calculations. If None, uses 5 convenient params and 4 modes (default None).

    Returns
    ----------
    output_dict : dict
        Dictionary that contains param name as a key and list of 2 values (MSE and R2 metrics) for that param.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if param_names is not None:
        num_pars_y = len(param_names)
    else:
        num_pars_y = 20

    y_log = torch.empty(size=[0, num_pars_y]).to(device)
    output_log = torch.empty(size=[0, num_pars_y]).to(device)

    for batch in data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        output_1, output_2 = model(x)
        # output = (output_1 * 0.5).add(output_2 * 0.5)
        output = output_2
        y_log = torch.cat((y_log, y), dim=0)
        output_log = torch.cat((output_log, output), dim=0)

    output_dict = calculate_metrics_torch(y_true=y_log, y_pred=output_log, param_names=param_names)

    return output_dict


@torch.inference_mode()
def calculate_val_metrics_branched_sep(model, data_loader, output_coeffs, param_names=None):
    """
    Calculates MSE and R2 metrics on a validation dataset for branched separate network.

    Parameters
    ----------
    model : model object
        Model that is being trained.
    data_loader : torch.utils.data.DataLoader
        Validation dataset dataloader.
    output_coeffs : list of float
        Should contain two float numbers which represent multiplier coefficients for short branch and long branches, correspondingly.
    param_names : list of str or None
        Parameter names (format `M{mode} Param_name`) to be used for metrics calculations. If None, uses 5 convenient params and 4 modes (default None).

    Returns
    ----------
    output_dict : dict
        Dictionary that contains param name as a key and list of 2 values (MSE and R2 metrics) for that param.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if param_names is not None:
        num_pars_y = len(param_names)
    else:
        num_pars_y = 20

    y_log = torch.empty(size=[0, num_pars_y]).to(device)
    output_log = torch.empty(size=[0, num_pars_y]).to(device)

    for batch in data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        output_1, output_2 = model(x)
        output = output_1 * output_coeffs[0] + output_2 * output_coeffs[1]
        y_log = torch.cat((y_log, y), dim=0)
        output_log = torch.cat((output_log, output), dim=0)

    output_dict = calculate_metrics_torch(y_true=y_log, y_pred=output_log, param_names=param_names)

    return output_dict


def train_mlp(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, param_names=None, plot_param='M1 Eigenfrequency (Hz)'):
    """
    Trains fully connected network.

    Parameters
    ----------
    model : model object
        Model that is being trained.
    train_loader : torch.utils.data.DataLoader
        Train dataset dataloader.
    val_loader : torch.utils.data.DataLoader
        Validation dataset dataloader.
    criterion : loss function object
    optimizer : optimizer object
    scheduler : scheduler object
    num_epochs : int
        Number of training epochs (default 100).
    param_names : list of str or None
        Parameter names (format `M{mode} Param_name`) to be used for metrics calculations. If None, uses 5 convenient params and 4 modes (default None).
    plot_param : str
        Parameter name for which MSE and R2 metrics will be plotted (default 'M1 Eigenfrequency (Hz)').

    Returns
    ----------
    pp : ProgressPlotter object
        Class object that contains history dict for the specified `plot_param`.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if param_names is not None:
        num_pars_y = len(param_names)
    else:
        num_pars_y = 20

    pp = ProgressPlotter()

    for epoch in range(num_epochs):
        y_log = torch.empty(size=[0, num_pars_y]).to(device)
        output_log = torch.empty(size=[0, num_pars_y]).to(device)

        model.train()
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y)

            y_log = torch.cat((y_log, y), dim=0)
            output_log = torch.cat((output_log, output), dim=0)

            loss.backward()
            optimizer.step()

        # Metric on train
        output_dict = calculate_metrics_torch(y_true=y_log, y_pred=output_log, param_names=param_names)

        # Logging
        if epoch % 10 == 0:
            pp.add_scalar('MSE_train', output_dict[plot_param][0].cpu().detach().numpy())
            pp.add_scalar('R2_train', output_dict[plot_param][1].cpu().detach().numpy())

        # Metrics on val
        model.eval()
        output_dict_val = calculate_val_metrics_mlp(model=model, data_loader=val_loader, param_names=param_names)

        scheduler.step(output_dict_val[plot_param][0])

        if epoch % 10 == 0:
            pp.add_scalar('MSE_val', output_dict_val[plot_param][0].cpu().detach().numpy())
            pp.add_scalar('R2_val', output_dict_val[plot_param][1].cpu().detach().numpy())

        if epoch % 10 == 0:
            pp.display([['MSE_train','MSE_val'], ['R2_train','R2_val']])
    return pp


def train_branched(model, train_loader, val_loader, criterion, optimizer, scheduler, loss_coeffs=None, num_epochs=100, param_names=None, plot_param='M1 Eigenfrequency (Hz)'):
    """
    Trains fully connected network.

    Parameters
    ----------
    model : model object
        Model that is being trained.
    train_loader : torch.utils.data.DataLoader
        Train dataset dataloader.
    val_loader : torch.utils.data.DataLoader
        Validation dataset dataloader.
    criterion : loss function object
    optimizer : optimizer object
    scheduler : scheduler object
    loss_coeffs : list of float
        List that contains loss multipliers (a_1 * loss_short_branch + a_2 * loss_long_branch) (default [0.2, 1]).
    num_epochs : int
        Number of training epochs (default 100).
    param_names : list of str or None
        Parameter names (format `M{mode} Param_name`) to be used for metrics calculations. If None, uses 5 convenient params and 4 modes (default None).
    plot_param : str
        Parameter name for which MSE and R2 metrics will be plotted (default 'M1 Eigenfrequency (Hz)').

    Returns
    ----------
    pp : ProgressPlotter object
        Class object that contains history dict for the specified `plot_param`.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if param_names is not None:
        num_pars_y = len(param_names)
    else:
        num_pars_y = 20

    if loss_coeffs is None:
        loss_coeffs = [0.2, 1]

    pp = ProgressPlotter()

    for epoch in range(num_epochs):
        y_log = torch.empty(size=[0, num_pars_y]).to(device)
        output_log = torch.empty(size=[0, num_pars_y]).to(device)

        model.train()
        for batch in train_loader:     # кусок данных для обучения
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            output_1, output_2 = model(x) # загружаем данные в модель
            loss_1 = criterion(output_1, y) # считаем ошибку
            loss_2 = criterion(output_2, y)
            loss = loss_coeffs[0] * loss_1 + loss_coeffs[1] * loss_2

            # output = (output_1 * 0.5).add(output_2 * 0.5) - uncomment if you want to make experiments with the model outputs
            output = output_2 # long branch is the output

            y_log = torch.cat((y_log, y), dim=0)
            output_log = torch.cat((output_log, output), dim=0)

            loss.backward()
            optimizer.step()

        # Берем все объекты из тренировочного даталоадера и считаем метрики
        output_dict = calculate_metrics_torch(y_true=y_log, y_pred=output_log, param_names=param_names)

        # Logging
        if epoch % 10 == 0:
            pp.add_scalar('MSE_train', output_dict[plot_param][0].cpu().detach().numpy())
            pp.add_scalar('R2_train', output_dict[plot_param][1].cpu().detach().numpy())

        model.eval()
        output_dict_val = calculate_val_metrics_branched(model=model, data_loader=val_loader, param_names=param_names)

        scheduler.step(output_dict_val[plot_param][0])

        if epoch % 10 == 0:
            pp.add_scalar('MSE_val', output_dict_val[plot_param][0].cpu().detach().numpy())
            pp.add_scalar('R2_val', output_dict_val[plot_param][1].cpu().detach().numpy())

        if epoch % 10 == 0:
            #pp.display([['loss_train', 'loss_val']])
            pp.display([['MSE_train','MSE_val'], ['R2_train','R2_val']])
    return pp


@torch.inference_mode()
def get_readable_metrics_mlp(model, data_loader, param_names=None):
    """
    Calculates MSE and R2 metrics on a dataset for fully connected network and converts it into readable format.

    Parameters
    ----------
    model : model object
        Model that is being trained.
    data_loader : torch.utils.data.DataLoader
        Validation dataset dataloader.
    param_names : list of str or None
        Parameter names (format `M{mode} Param_name`) to be used for metrics calculations. If None, uses 5 convenient params and 4 modes (default None).

    Returns
    ----------
    output_dict : dict
        Dictionary that contains param name as a key and list of 2 values (MSE and R2 metrics) for that param. For instance, {'M1 Eigenfrequency (Hz)': [0.01, 0.95]}
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if param_names is not None:
        num_pars_y = len(param_names)
    else:
        num_pars_y = 20

    y_log = torch.empty(size=[0, num_pars_y]).to(device)
    output_log = torch.empty(size=[0, num_pars_y]).to(device)

    for batch in data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        output = model(x)
        y_log = torch.cat((y_log, y), dim=0)
        output_log = torch.cat((output_log, output), dim=0)

    output_dict = calculate_metrics_torch(y_true=y_log, y_pred=output_log, param_names=param_names)

    for name in output_dict.keys():
        for index in range(0, len(output_dict[name])):
            output_dict[name][index] = output_dict[name][index].item()

    return output_dict
