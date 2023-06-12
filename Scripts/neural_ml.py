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


def get_elapsed_time_mlp(model, data_loader, num_samples=200, param_names_x=None):
    """
    Returns time elapsed during evaluation using fully connected network.

    Parameters
    ----------
    model : model object
        Trained model evaluation time to be measured for.
    data_loader : torch.utils.data.DataLoader
        Validation dataset dataloader.
    num_samples : int
        Number of samples to be used for evaluation (default 200).
    param_names_x : list of str or None
        X-parameter names (format `Parameter`) to be used for metrics calculations. If None, uses 8 convenient params (default None).

    Returns
    ----------
    elapsed_time : float
        Elapsed time.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if param_names_x is not None:
        num_pars_x = len(param_names_x)
    else:
        param_names_x = ['Beam length (um)', 'Beam width (nm)',
                         'Thickness_1 (nm)', 'Thickness_2 (nm)',
                         'Temperature (K)', 'Distance (nm)',
                         'Gate voltage (V)', 'Pretension (Pa)']
        num_pars_x = len(param_names_x)

    samples_x = torch.empty(size=[0, num_pars_x]).to(device)
    for batch in data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        samples_x = torch.cat((samples_x, x), dim=0)

        if samples_x.shape[0] > num_samples:
            break

    samples_x = samples_x[:200, :]

    time_start = time()
    _ = model(samples_x)
    time_stop = time()

    elapsed_time = time_stop - time_start

    return elapsed_time


def get_elapsed_time_branched(model, data_loader, num_samples=200, param_names_x=None):
    """
    Returns time elapsed during evaluation using branched fully connected network.

    Parameters
    ----------
    model : model object
        Trained model evaluation time to be measured for.
    data_loader : torch.utils.data.DataLoader
        Validation dataset dataloader.
    num_samples : int
        Number of samples to be used for evaluation (default 200).
    param_names_x : list of str or None
        X-parameter names (format `Parameter`) to be used for metrics calculations. If None, uses 8 convenient params (default None).

    Returns
    ----------
    elapsed_time : float
        Elapsed time.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if param_names_x is not None:
        num_pars_x = len(param_names_x)
    else:
        param_names_x = ['Beam length (um)', 'Beam width (nm)',
                         'Thickness_1 (nm)', 'Thickness_2 (nm)',
                         'Temperature (K)', 'Distance (nm)',
                         'Gate voltage (V)', 'Pretension (Pa)']
        num_pars_x = len(param_names_x)

    samples_x = torch.empty(size=[0, num_pars_x]).to(device)
    for batch in data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        samples_x = torch.cat((samples_x, x), dim=0)

        if samples_x.shape[0] > num_samples:
            break

    samples_x = samples_x[:200, :]

    time_start = time()
    _, _ = model(samples_x)
    time_stop = time()

    elapsed_time = time_stop - time_start

    return elapsed_time


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
        self._history_dict[tag].append(value)

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
            ax.plot(self._history_dict[key], marker=".", label=key)
            history_len = max(history_len, len(self.history_dict[key]))
        if len(tags) > 1:
            ax.legend(loc="lower left")
        else:
            ax.set_ylabel(key)
        ax.set_xlabel('Epoch')
        ax.set_xticks(np.arange(0, history_len, 10))
        ax.set_xticklabels(np.arange(0, history_len, 10))

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


@torch.inference_mode()
def calculate_val_metrics_tandem(model_inverse, model_forward, data_loader, param_names_x=None, param_names_y=None):
    """
    Calculates MSE and R2 metrics on a validation dataset for tandem network.

    Parameters
    ----------
    model_inverse : model object
        Inverse tandem model.
    model_forward : model object
        Forward tandem model.
    data_loader : torch.utils.data.DataLoader
        Validation dataset dataloader.
    param_names_x : list of str or None
        X-parameter names (format `Parameter`) to be used for metrics calculations. If None, uses 8 convenient params (default None).
    param_names_y : list of str or None
        Y-parameter names (format `M{mode} Param_name`) to be used for metrics calculations. If None, uses 5 convenient params and 4 modes (default None).

    Returns
    ----------
    output_dict_inverse : dict
        Dictionary that contains MSE and R2 metrics for the inverse model output (e.g. X-data).
    output_dict_forward : dict
        Dictionary that contains MSE and R2 metrics for the forward model output (e.g. Y-data).
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if param_names_x is not None:
        num_pars_x = len(param_names_x)
    else:
        param_names_x = ['Beam length (um)', 'Beam width (nm)',
                         'Thickness_1 (nm)', 'Thickness_2 (nm)',
                         'Temperature (K)', 'Distance (nm)',
                         'Gate voltage (V)', 'Pretension (Pa)']
        num_pars_x = len(param_names_x)

    if param_names_y is not None:
        num_pars_y = len(param_names_y)
    else:
        num_pars_y = 20

    # Logs for X-data
    x_log = torch.empty(size=[0, num_pars_x]).to(device)
    output_inverse_log = torch.empty(size=[0, num_pars_x]).to(device)

    # Logs for Y-data
    y_log = torch.empty(size=[0, num_pars_y]).to(device)
    output_forward_log = torch.empty(size=[0, num_pars_y]).to(device)

    model_inverse.eval()
    model_forward.eval()

    for batch in data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        # Obtaining predictions of inverse and then forward model predictions
        output_inverse = model_inverse(y)
        output_forward = model_forward(output_inverse)

        x_log = torch.cat((x_log, x), dim=0)
        y_log = torch.cat((y_log, y), dim=0)
        output_inverse_log = torch.cat((output_inverse_log, output_inverse), dim=0)
        output_forward_log = torch.cat((output_forward_log, output_forward), dim=0)

    output_dict_inverse = calculate_metrics_torch(y_true=x_log, y_pred=output_inverse_log, param_names=param_names_x)
    output_dict_forward = calculate_metrics_torch(y_true=y_log, y_pred=output_forward_log, param_names=param_names_y)

    return output_dict_inverse, output_dict_forward


@torch.inference_mode()
def calculate_val_metrics_tandem_cond(model_inverse_cond, model_forward, data_loader, fix_indices, param_names_x=None, param_names_y=None):
    """
    Calculates MSE and R2 metrics on a validation dataset for conditional tandem network.

    Parameters
    ----------
    model_inverse_cond : model object
        Inverse tandem model that, in addition to convenient Y-data input, has 4 inputs for fixed X-data (temp., distance, voltage and pretension).
    model_forward : model object
        Forward tandem model.
    data_loader : torch.utils.data.DataLoader
        Validation dataset dataloader.
    fix_indices : list of int
        List that contains indices of fixed parameters in param_names_x.
    param_names_x : list of str or None
        X-parameter names (format `Parameter`) to be used for metrics calculations. If None, uses 8 convenient params (default None).
    param_names_y : list of str or None
        Y-parameter names (format `M{mode} Param_name`) to be used for metrics calculations. If None, uses 5 convenient params and 4 modes (default None).

    Returns
    ----------
    output_dict_inverse : dict
        Dictionary that contains MSE and R2 metrics for the inverse model output (e.g. X-data).
    output_dict_forward : dict
        Dictionary that contains MSE and R2 metrics for the forward model output (e.g. Y-data).
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if param_names_x is not None:
        num_pars_x = len(param_names_x)
    else:
        param_names_x = ['Beam length (um)', 'Beam width (nm)',
                         'Thickness_1 (nm)', 'Thickness_2 (nm)',
                         'Temperature (K)', 'Distance (nm)',
                         'Gate voltage (V)', 'Pretension (Pa)']
        num_pars_x = len(param_names_x)

    if param_names_y is not None:
        num_pars_y = len(param_names_y)
    else:
        num_pars_y = 20

    # Getting non-fixed indices
    nonfix_indices = [index for index in range(0, num_pars_x) if index not in fix_indices]

    # Logs for X-data
    x_log = torch.empty(size=[0, num_pars_x]).to(device)
    output_inverse_log = torch.empty(size=[0, num_pars_x]).to(device)

    # Logs for Y-data
    y_log = torch.empty(size=[0, num_pars_y]).to(device)
    output_forward_log = torch.empty(size=[0, num_pars_y]).to(device)

    model_inverse_cond.eval()
    model_forward.eval()

    for batch in data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        x_fix = x[:, fix_indices]   # choose conditional variables

        # Logging
        x_log = torch.cat((x_log, x), dim=0)
        y_log = torch.cat((y_log, y), dim=0)

        # Put resonant parameters and fixed parameters into the inverse model
        output_inverse = model_inverse_cond(y, x_fix)

        # In output_inverse we have only non-fixed x-parameters --> we need to replace them in `x` in order to input `x` into the forward model
        x[:, nonfix_indices] = output_inverse

        # Put x-parameters into the forward model
        output_forward = model_forward(x)

        # Logging
        output_inverse_log = torch.cat((output_inverse_log, x), dim=0)
        output_forward_log = torch.cat((output_forward_log, output_forward), dim=0)

    output_dict_inverse = calculate_metrics_torch(y_true=x_log, y_pred=output_inverse_log, param_names=param_names_x)
    output_dict_forward = calculate_metrics_torch(y_true=y_log, y_pred=output_forward_log, param_names=param_names_y)

    return output_dict_inverse, output_dict_forward


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

    for epoch in range(num_epochs+1):
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
        pp.add_scalar('MSE_train', output_dict[plot_param][0].cpu().detach().numpy())
        pp.add_scalar('R2_train', output_dict[plot_param][1].cpu().detach().numpy())

        # Metrics on val
        model.eval()
        output_dict_val = calculate_val_metrics_mlp(model=model, data_loader=val_loader, param_names=param_names)

        scheduler.step(output_dict_val[plot_param][0])

        # Logging
        pp.add_scalar('MSE_val', output_dict_val[plot_param][0].cpu().detach().numpy())
        pp.add_scalar('R2_val', output_dict_val[plot_param][1].cpu().detach().numpy())

        if epoch % 10 == 0:
            pp.display([['MSE_train', 'MSE_val'], ['R2_train', 'R2_val']])
    return pp


def train_branched(model,
                   train_loader, val_loader,
                   criterion, optimizer, scheduler,
                   loss_coeffs=None, num_epochs=100, param_names=None,
                   plot_param='M1 Eigenfrequency (Hz)'):
    """
    Trains branched network.

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

    for epoch in range(num_epochs+1):
        y_log = torch.empty(size=[0, num_pars_y]).to(device)
        output_log = torch.empty(size=[0, num_pars_y]).to(device)

        model.train()
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            output_1, output_2 = model(x)
            loss_1 = criterion(output_1, y)
            loss_2 = criterion(output_2, y)
            loss = loss_coeffs[0] * loss_1 + loss_coeffs[1] * loss_2

            # output = (output_1 * 0.5).add(output_2 * 0.5) - uncomment if you want to make experiments with the model outputs
            output = output_2  # long branch is the output

            y_log = torch.cat((y_log, y), dim=0)
            output_log = torch.cat((output_log, output), dim=0)

            loss.backward()
            optimizer.step()

        # Берем все объекты из тренировочного даталоадера и считаем метрики
        output_dict = calculate_metrics_torch(y_true=y_log, y_pred=output_log, param_names=param_names)

        # Logging
        pp.add_scalar('MSE_train', output_dict[plot_param][0].cpu().detach().numpy())
        pp.add_scalar('R2_train', output_dict[plot_param][1].cpu().detach().numpy())

        model.eval()
        output_dict_val = calculate_val_metrics_branched(model=model, data_loader=val_loader, param_names=param_names)

        scheduler.step(output_dict_val[plot_param][0])
        
        # Logging
        pp.add_scalar('MSE_val', output_dict_val[plot_param][0].cpu().detach().numpy())
        pp.add_scalar('R2_val', output_dict_val[plot_param][1].cpu().detach().numpy())

        if epoch % 10 == 0:
            pp.display([['MSE_train', 'MSE_val'], ['R2_train', 'R2_val']])
    return pp


# train_branched_sep


def train_tandem(model_inverse, model_forward,
                 train_loader, val_loader,
                 criterion, optimizer, scheduler,
                 num_epochs=100, param_names_x=None, param_names_y=None,
                 plot_param='M1 Eigenfrequency (Hz)'):
    """
    Trains inverse model in tandem network.

    Parameters
    ----------
    model_inverse : model object
        Inverse tandem model.
    model_forward : model object
        Forward tandem model.
    train_loader : torch.utils.data.DataLoader
        Train dataset dataloader.
    val_loader : torch.utils.data.DataLoader
        Validation dataset dataloader.
    criterion : loss function object
    optimizer : optimizer object
    scheduler : scheduler object
    num_epochs : int
        Number of training epochs (default 100).
    param_names_x : list of str or None
        X-parameter names (format `Parameter`) to be used for metrics calculations. If None, uses 8 convenient params (default None).
    param_names_y : list of str or None
        Y-parameter names (format `M{mode} Param_name`) to be used for metrics calculations. If None, uses 5 convenient params and 4 modes (default None).
    plot_param : str
        Parameter name for which MSE and R2 metrics will be plotted (default 'M1 Eigenfrequency (Hz)').

    Returns
    ----------
    pp : ProgressPlotter object
        Class object that contains history dict for the specified `plot_param`.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if param_names_x is not None:
        num_pars_x = len(param_names_x)
    else:
        param_names_x = ['Beam length (um)', 'Beam width (nm)',
                         'Thickness_1 (nm)', 'Thickness_2 (nm)',
                         'Temperature (K)', 'Distance (nm)',
                         'Gate voltage (V)', 'Pretension (Pa)']
        num_pars_x = len(param_names_x)

    if param_names_y is not None:
        num_pars_y = len(param_names_y)
    else:
        num_pars_y = 20

    pp = ProgressPlotter()

    model_forward.eval()

    for epoch in range(num_epochs+1):

        model_inverse.train()

        for batch in train_loader:  # Training inverse model
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Obtaining predictions of inverse and then forward model predictions
            output_inverse = model_inverse(y)

            output_forward = model_forward(output_inverse)

            loss = criterion(output_forward, y)  # calculating loss
            #loss += criterion(output_inverse, x)  # uncomment if you want to add loss on X-data
            loss.backward()
            optimizer.step()

        # Calculating MSE and R2 metrics on train and val loaders
        with torch.no_grad():
            output_dict_train_inverse, output_dict_train_forward = calculate_val_metrics_tandem(model_inverse, model_forward,
                                                                                                train_loader, param_names_x, param_names_y)
            output_dict_val_inverse, output_dict_val_forward = calculate_val_metrics_tandem(model_inverse, model_forward,
                                                                                            val_loader, param_names_x, param_names_y)

        # Scheduler step
        if plot_param in param_names_x:
            scheduler.step(output_dict_val_inverse[plot_param][0])
        elif plot_param in param_names_y:
            scheduler.step(output_dict_val_forward[plot_param][0])

        # Logging
        if plot_param in param_names_x:
            pp.add_scalar('MSE_train', output_dict_train_inverse[plot_param][0].cpu().detach().numpy())
            pp.add_scalar('R2_train', output_dict_train_inverse[plot_param][1].cpu().detach().numpy())
            pp.add_scalar('MSE_val', output_dict_val_inverse[plot_param][0].cpu().detach().numpy())
            pp.add_scalar('R2_val', output_dict_val_inverse[plot_param][1].cpu().detach().numpy())

        elif plot_param in param_names_y:
            pp.add_scalar('MSE_train', output_dict_train_forward[plot_param][0].cpu().detach().numpy())
            pp.add_scalar('R2_train', output_dict_train_forward[plot_param][1].cpu().detach().numpy())
            pp.add_scalar('MSE_val', output_dict_val_forward[plot_param][0].cpu().detach().numpy())
            pp.add_scalar('R2_val', output_dict_val_forward[plot_param][1].cpu().detach().numpy())

        # Displaying current results
        if epoch % 10 == 0:
            pp.display([['MSE_train', 'MSE_val'], ['R2_train', 'R2_val']])
    return pp


def train_tandem_cond(model_inverse_cond, model_forward,
                      fix_params,
                      train_loader, val_loader,
                      criterion, optimizer, scheduler,
                      num_epochs=100, param_names_x=None, param_names_y=None,
                      plot_param='M1 Eigenfrequency (Hz)'):
    """
    Trains conditional inverse model in tandem network.

    Sometimes when solving the reverse problem (find input by output) it is required to have some of the input parameters to be fixed. These are called conditional ones.

    Parameters
    ----------
    model_inverse_cond : model object
        Inverse tandem model that, in addition to convenient Y-data input, has some inputs fixed (e.g. temperature, distance, voltage and pretension).
    model_forward : model object
        Forward tandem model.
    fix_params : list of str
        List that contains param names which are to be fixed.
    train_loader : torch.utils.data.DataLoader
        Train dataset dataloader.
    val_loader : torch.utils.data.DataLoader
        Validation dataset dataloader.
    criterion : loss function object
    optimizer : optimizer object
    scheduler : scheduler object
    num_epochs : int
        Number of training epochs (default 100).
    param_names_x : list of str or None
        X-parameter names (format `Parameter`) to be used for metrics calculations. If None, uses 8 convenient params (default None).
    param_names_y : list of str or None
        Y-parameter names (format `M{mode} Param_name`) to be used for metrics calculations. If None, uses 5 convenient params and 4 modes (default None).
    plot_param : str
        Parameter name for which MSE and R2 metrics will be plotted (default 'M1 Eigenfrequency (Hz)').

    Returns
    ----------
    pp : ProgressPlotter object
        Class object that contains history dict for the specified `plot_param`.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if param_names_x is not None:
        num_pars_x = len(param_names_x)
    else:
        param_names_x = ['Beam length (um)', 'Beam width (nm)',
                         'Thickness_1 (nm)', 'Thickness_2 (nm)',
                         'Temperature (K)', 'Distance (nm)',
                         'Gate voltage (V)', 'Pretension (Pa)']
        num_pars_x = len(param_names_x)

    if param_names_y is not None:
        num_pars_y = len(param_names_y)
    else:
        num_pars_y = 20
    
    fix_indices = get_fix_indices(param_names_x, fix_params)

    # Getting non-fixed indices
    nonfix_indices = [index for index in range(0, num_pars_x) if index not in fix_indices]

    pp = ProgressPlotter()

    model_forward.eval()

    for epoch in range(num_epochs+1):

        model_inverse_cond.train()

        for batch in train_loader:  # Training inverse model
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Fixed x-parameters
            x_fix = x[:, fix_indices]

            # Non-fixed x-parameters (they're required to calculate loss below)
            x_nonfix_true = x[:, nonfix_indices]
            optimizer.zero_grad()

            # Obtaining predictions of inverse model
            output_inverse = model_inverse_cond(y, x_fix)

            # In output_inverse we have only non-fixed x-parameters --> we need to replace them in `x` in order to input `x` into the forward model
            x[:, nonfix_indices] = output_inverse

            output_forward = model_forward(x)

            loss = criterion(output_forward, y)  # calculating loss
            loss += criterion(output_inverse, x_nonfix_true)
            loss.backward()
            optimizer.step()

        # Calculating MSE and R2 metrics on train and val loaders
        with torch.no_grad():
            output_dict_train_inverse, output_dict_train_forward = calculate_val_metrics_tandem_cond(model_inverse_cond,
                                                                                                     model_forward,
                                                                                                     train_loader,
                                                                                                     fix_indices,
                                                                                                     param_names_x,
                                                                                                     param_names_y)
            output_dict_val_inverse, output_dict_val_forward = calculate_val_metrics_tandem_cond(model_inverse_cond,
                                                                                                 model_forward,
                                                                                                 val_loader,
                                                                                                 fix_indices,
                                                                                                 param_names_x,
                                                                                                 param_names_y)

        # Scheduler step
        if plot_param in param_names_x:
            scheduler.step(output_dict_val_inverse[plot_param][0])
        elif plot_param in param_names_y:
            scheduler.step(output_dict_val_forward[plot_param][0])

        # Logging
        if plot_param in param_names_x:
            pp.add_scalar('MSE_train', output_dict_train_inverse[plot_param][0].cpu().detach().numpy())
            pp.add_scalar('R2_train', output_dict_train_inverse[plot_param][1].cpu().detach().numpy())
            pp.add_scalar('MSE_val', output_dict_val_inverse[plot_param][0].cpu().detach().numpy())
            pp.add_scalar('R2_val', output_dict_val_inverse[plot_param][1].cpu().detach().numpy())

        elif plot_param in param_names_y:
            pp.add_scalar('MSE_train', output_dict_train_forward[plot_param][0].cpu().detach().numpy())
            pp.add_scalar('R2_train', output_dict_train_forward[plot_param][1].cpu().detach().numpy())
            pp.add_scalar('MSE_val', output_dict_val_forward[plot_param][0].cpu().detach().numpy())
            pp.add_scalar('R2_val', output_dict_val_forward[plot_param][1].cpu().detach().numpy())

        # Displaying current results
        if epoch % 10 == 0:
            pp.display([['MSE_train', 'MSE_val'], ['R2_train', 'R2_val']])
    return pp


def get_fix_indices(param_names, fix_params):
    """
    Returns indices of specified parameter names in the list.
    
    Parameters
    ----------
    param_names : list of str
        List containing all param names.
    fix_params : list of str or None
        List containing fixed param names.

    Returns
    ----------
    fix_indices : list of int
        Indies of fix_params elements occurencies in param_names list.
    """
    fix_indices = [i for i, x in enumerate(param_names) if x in fix_params]

    return fix_indices
        

def train_branched_wrapped(wrapper,
                           train_loader, val_loader,
                           criterion, optimizer, scheduler,
                           num_epochs=100, param_names=None,
                           plot_param='M1 Eigenfrequency (Hz)'):
    """
    Trains branch coefficients and neural model in a wrapper model.

    Parameters
    ----------
    wrapper : wrapper object
        Instance of class that contains both trainable loss coefficients and trainable branched model.
    train_loader : torch.utils.data.DataLoader
        Train dataset dataloader.
    val_loader : torch.utils.data.DataLoader
        Validation dataset dataloader.
    criterion : loss function object
    optimizer : optimizer object
    scheduler : scheduler object
    branch_num : int
        Number of branches in neural model inside the wrapper (default 2 as for Branched MLP model).
    num_epochs : int
        Number of training epochs (default 100).
    param_names : list of str or None
        Y-parameter names (format `Parameter`) to be used for metrics calculations. If None, uses 8 convenient params (default None).
    plot_param : str
        Parameter name for which MSE and R2 metrics will be plotted (default 'M1 Eigenfrequency (Hz)').

    Returns
    ----------
    pp : ProgressPlotter object
        Class object that contains history dict for the specified `plot_param`.
    loss_coeffs : list of float with len = branch_num
        Optimal loss coefficients.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if param_names is not None:
        num_pars_y = len(param_names)
    else:
        num_pars_y = 20

    pp = ProgressPlotter()

    for epoch in range(num_epochs+1):
        y_log = torch.empty(size=[0, num_pars_y]).to(device)
        output_log = torch.empty(size=[0, num_pars_y]).to(device)

        wrapper.train()
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            loss, loss_coeffs = wrapper(input=x, target=y)

            with torch.no_grad():
                output_1, output_2 = wrapper.model(x)  # obtain results for logging

            output = output_2  # long branch is treated as the model output

            y_log = torch.cat((y_log, y), dim=0)
            output_log = torch.cat((output_log, output), dim=0)

            loss.backward()
            optimizer.step()

        # Metrics on val_loader
        output_dict = calculate_metrics_torch(y_true=y_log, y_pred=output_log, param_names=param_names)

        # Logging
        pp.add_scalar('MSE_train', output_dict[plot_param][0].cpu().detach().numpy())
        pp.add_scalar('R2_train', output_dict[plot_param][1].cpu().detach().numpy())

        wrapper.eval()
        output_dict_val = calculate_val_metrics_branched(wrapper.model, val_loader, param_names)

        scheduler.step(output_dict_val[plot_param][0])

        # Logging
        pp.add_scalar('MSE_val', output_dict_val[plot_param][0].cpu().detach().numpy())
        pp.add_scalar('R2_val', output_dict_val[plot_param][1].cpu().detach().numpy())

        if epoch % 10 == 0:
            pp.display([['MSE_train', 'MSE_val'], ['R2_train', 'R2_val']])
    return pp, loss_coeffs


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

    output_dict = calculate_val_metrics_mlp(model, data_loader, param_names)

    for name in output_dict.keys():
        for index in range(0, len(output_dict[name])):
            output_dict[name][index] = output_dict[name][index].item()

    return output_dict


@torch.inference_mode()
def get_readable_metrics_branched(model, data_loader, param_names=None):
    """
    Calculates MSE and R2 metrics on a dataset for branched network and converts it into readable format.

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
    output_dict = calculate_val_metrics_branched(model, data_loader, param_names)
    
    for name in output_dict.keys():
        for index in range(0, len(output_dict[name])):
            output_dict[name][index] = output_dict[name][index].item()

    return output_dict


@torch.inference_mode()
def get_readable_metrics_tandem(model_inverse, model_forward, data_loader, param_names_x=None, param_names_y=None):
    """
    Calculates MSE and R2 metrics on a dataset for branched network and converts it into readable format.

    Parameters
    ----------
    model_inverse : model object
        Inverse tandem model.
    model_forward : model object
        Forward tandem model.
    data_loader : torch.utils.data.DataLoader
        Validation dataset dataloader.
    param_names_x : list of str or None
        X-parameter names (format `Parameter`) to be used for metrics calculations. If None, uses 8 convenient params (default None).
    param_names_y : list of str or None
        Y-parameter names (format `M{mode} Param_name`) to be used for metrics calculations. If None, uses 5 convenient params and 4 modes (default None).

    Returns
    ----------
    output_dict_inverse : dict
        Dictionary that contains x-param name as a key and list of 2 values (MSE and R2 metrics) for that x-param. For instance, {'Beam lenght (um)': [0.01, 0.95]}.
    output_dict_forward : dict
        Dictionary that contains y-param name as a key and list of 2 values (MSE and R2 metrics) for that y-param. For instance, {'M1 Eigenfrequency (Hz)': [0.01, 0.95]}
    
    """
    output_dict_inverse, output_dict_forward = calculate_val_metrics_tandem(model_inverse, model_forward,
                                                                            data_loader, param_names_x, param_names_y)
    
    for name in output_dict_inverse.keys():
        for index in range(0, len(output_dict_inverse[name])):
            output_dict_inverse[name][index] = output_dict_inverse[name][index].item()
    
    for name in output_dict_forward.keys():
        for index in range(0, len(output_dict_forward[name])):
            output_dict_forward[name][index] = output_dict_forward[name][index].item()

    return output_dict_inverse, output_dict_forward


@torch.inference_mode()
def get_readable_metrics_tandem_cond(model_inverse_cond, model_forward, data_loader, fix_indices, param_names_x=None, param_names_y=None):
    """
    Calculates MSE and R2 metrics on a dataset for branched network and converts it into readable format.

    Parameters
    ----------
    model_inverse_cond : model object
        Inverse tandem model that, in addition to convenient Y-data input, has some inputs fixed (e.g. temperature, distance, voltage and pretension).
    model_forward : model object
        Forward tandem model.
    data_loader : torch.utils.data.DataLoader
        Validation dataset dataloader.
    param_names_x : list of str or None
        X-parameter names (format `Parameter`) to be used for metrics calculations. If None, uses 8 convenient params (default None).
    param_names_y : list of str or None
        Y-parameter names (format `M{mode} Param_name`) to be used for metrics calculations. If None, uses 5 convenient params and 4 modes (default None).

    Returns
    ----------
    output_dict_inverse : dict
        Dictionary that contains x-param name as a key and list of 2 values (MSE and R2 metrics) for that x-param. For instance, {'Beam lenght (um)': [0.01, 0.95]}.
    output_dict_forward : dict
        Dictionary that contains y-param name as a key and list of 2 values (MSE and R2 metrics) for that y-param. For instance, {'M1 Eigenfrequency (Hz)': [0.01, 0.95]}
    
    """
    output_dict_inverse, output_dict_forward = calculate_val_metrics_tandem_cond(model_inverse_cond,
                                                                                 model_forward,
                                                                                 data_loader,
                                                                                 fix_indices,
                                                                                 param_names_x,
                                                                                 param_names_y)
    
    for name in output_dict_inverse.keys():
        for index in range(0, len(output_dict_inverse[name])):
            output_dict_inverse[name][index] = output_dict_inverse[name][index].item()
    
    for name in output_dict_forward.keys():
        for index in range(0, len(output_dict_forward[name])):
            output_dict_forward[name][index] = output_dict_forward[name][index].item()

    return output_dict_inverse, output_dict_forward


def compare_models(dict_list, model_names, param_names, apply_log_mse, apply_log_r2, modes=None, sharey='row'):
    """
    Plots metrics of specified models on one plot.

    Parameters
    ----------
    dict_list : list of dict
        List that contains dictionaries of MSE and R2 metrics for all the models in `model_names` (order must be the same).
    model_names : list of str
        List that contains names of models (order must be the same).
    param_names : list of str or None
        Parameter names (format `M{mode} Param_name`) to be used for plotting metrics. If None, uses 5 convenient params and 4 modes (default None).
    apply_log_mse : bool
        If True, y-axis of all MSE metric plots will be log-scaled.
    apply_log_r2 : bool
        If True, y-axis of all R2 metric plots will be log-scaled.
    modes : list of int
        List that contains specified numbers of modes to be plotted (default [1, 2, 3, 4]).
    sharey : bool or {'none', 'all', 'row', 'col'} (default 'row').
        Controls sharing of properties among x (*sharex*) or y (*sharey*) axes:
        - True or 'all': x- or y-axis will be shared among all subplots.
        - False or 'none': each subplot x- or y-axis will be independent.
        - 'row': each subplot row will share an x- or y-axis.
        - 'col': each subplot column will share an x- or y-axis.
    """
    if modes is None:
        modes = [1, 2, 3, 4]

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(25, 10), sharey=sharey)

    for row_index in range(0, 2):

        for col_index in range(0, 5):

            for model_index in range(0, len(model_names)):
                points_to_plot = []
                for mode in modes:
                    points_to_plot.append(dict_list[model_index][f'M{mode} {param_names[col_index]}'][row_index])

                ax[row_index, col_index].plot(modes, points_to_plot, label=model_names[model_index])

            # MSE metrics postprocessing
            if row_index == 0:
                if apply_log_mse:
                    ax[row_index, col_index].set_yscale('log')

                ax[row_index, col_index].set_ylabel('MSE Loss')
                ax[row_index, col_index].set_title(f"{param_names[col_index]} MSE Loss", fontsize=2 * 6)

            # R2 metrics postprocessing
            if row_index == 1:
                if apply_log_r2:
                    ax[row_index, col_index].set_yscale('log')

                ax[row_index, col_index].set_ylabel('R2 Loss')
                ax[row_index, col_index].set_title(f"{param_names[col_index]} R2 Score", fontsize=2 * 6)

            ax[row_index, col_index].set_xlabel('Mode number')
            ax[row_index, col_index].tick_params(axis='y', labelleft=True)
            ax[row_index, col_index].set_xticks(modes)
            ax[row_index, col_index].grid(visible=True)
            ax[row_index, col_index].legend()
    plt.tight_layout()
    plt.plot()
