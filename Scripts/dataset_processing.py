import pandas as pd
import numpy as np


def row_col_transform(df, num_common):
    """
    Converts the whole dataset from the form of:
    geometry_params_1, resonant_mode_1
    geometry_params_1, resonant_mode_2
    geometry_params_1, resonant_mode_1
    geometry_params_1, resonant_mode_2

    to the form of:
    geometry_params_1, resonant_mode_1, resonant_mode_2, resonant_mode_3, resonant_mode_4

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be processed.
    num_common : int
        Number of parameters which are general for all resonant modes, should go first in df.columns.

    Returns
    ----------
    df : pd.DataFrame
        Processed dataframe with unnamed columns
    """
    # Here we save rows from initial DataFrame:
    data = []

    # Iterating through all rows of the initial DataFrame:
    for index in range(0, df.shape[0]):

        # Choose every 4-th row in the initial DataFrame:
        if index % 4 == 0:

            # Iterator variable that makes counts of resonant modes
            i = 0

            # Extracting row in the form of 'geometry_params_1, resonant_mode_1'
            row = df.iloc[index].values.flatten().tolist()

        else:
            # Extracting row in the form of 'resonant_mode_(i+1)'
            row += df.iloc[index, num_common:].values.flatten().tolist()

        i += 1

        # If we have taken all four rows we append it to the output ('data' list):
        if i == 4:
            data.append(row)

    return pd.DataFrame(data)



def name_columns(df, data, num_common):
    """
    Names columns in 'data' DataFrame as in 'df' DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with named columns
    data : pd.DataFrame
        Dataframe which has new format (see function row_col_transform) but has unnamed columns
    num_common : int
        Number of parameters which are common for all resonant modes, should go first in df.columns.

    Returns
    ----------
    data : pd.DataFrame
        Processed dataframe with named columns
    """
    # Number of different mode columns (all columns minus common columns)
    num_differ = df.shape[1] - num_common

    cols_common = list(df.columns)[:num_common]
    cols_differ = list(df.columns)[num_common:]

    # List where new names of columns will be saved
    columns = []

    # Saving names of common columns
    for col_index in range(0, num_common):
        columns.append(cols_common[col_index])

    # Saving names of different mode columns
    for mode_index in range(1, 5):
        for col_index in range(0, num_differ):
            columns.append(f'M{mode_index} ' + cols_differ[col_index])

    data.columns = columns

    return data


def return_neg_frequencies_index(df):
    """
    Returns row and column indices which contain resonant modes with Im(f) < 0.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be checked for Im(f) < 0 anomaly.

    Returns
    ----------
    anomaly_rows : list of int
        List of row indices that contain Im(f) < 0.
    anomaly_cols : list of int
        List of column indices that contain Im(f) < 0.
    """
    # List containing indices of resonant frequency columns
    freq_indices = []
    for mode in range(1, 5):
        freq_indices.append(df.columns.get_loc(f'M{mode} ' + 'Eigenfrequency (Hz)'))

    anomaly_rows = []
    anomaly_cols = []
    for col_index in freq_indices:
        for row_index in range(df.shape[0]):
            if '-' in df.iloc[row_index, col_index]:
                anomaly_rows.append(row_index)
                anomaly_cols.append(col_index)
    return anomaly_rows, anomaly_cols


def del_frequencies(df, anomaly_rows):
    """
    Deletes rows containing at least one resonant frequency with Im(f) < 0 (and, optionally, other anomalies if column indices are known).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing rows with anomalies.
    anomaly_rows : list of int
        List containing row indices where anomaly was detected.

    Returns
    ----------
    df : pd.DataFrame
        Processed dataframe with anomaly rows deleted.
    """
    # Making sure we account a row only once
    anomaly_rows = [x for x in set(anomaly_rows)]

    # It is easier to delete rows from the end since this operation doesn't change index of other rows
    anomaly_rows.sort(reverse=True)
    for row_index in anomaly_rows:
        df.drop(index=row_index, inplace=True)

    return df


def del_im_frequency(df):
    """
    Deletes imaginary part of every resonant frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be processed.

    Returns
    ----------
    df : pd.DataFrame
        Dataframe with imaginary part of resonant frequencies deleted.
    """
    # List containing indices of resonant frequency columns
    freq_indices = []
    for mode in range(1, 5):
        freq_indices.append(df.columns.get_loc(f'M{mode} ' + 'Eigenfrequency (Hz)'))

    for col_index in freq_indices:
        for row_index in range(0, df.shape[0]):
            df.iloc[row_index, col_index] = float(str(df.iloc[row_index, col_index]).split('+', 1)[0])
    return df


def catch_limit_overflow(df, param_name, lower_limit, upper_limit):
    """
    Masks dataframe by limiting specified parameter between lower and upper limits.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be masked.
    param_name : str
        Parameter name to be limited. Note: it applies limit to all resonant modes.
    lower_limit : float
        Lower limit for the specified parameter.
    upper_limit : float
        Upper limit for the specified parameter.

    Returns
    ----------
    df: pd.DataFrame
        Masked dataframe.
    """
    mask = (df[param_name].astype(float) > lower_limit) & (df[param_name].astype(float) < upper_limit)
    return df[mask]


def set_param_limits(df, param_limits):
    """
    Sets limits for specified parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe which parameters are to be limited.
    param_limits : dict
        Dictionary that contains parameters to be limited as keys and list of two values (lower and upper limit) as values. For instance, param_limits = {'Eigenfrequency (Hz)': [0, 1e7]}.

    Returns
    ----------
    df : pd.DataFrame
        Dataframe with specified parameters limited.
    """
    # Converts resonant frequency columns to `float`
    for mode_number in range(1, 5):
        df.loc[:, f'M{mode_number} ' + 'Eigenfrequency (Hz)'] = df.loc[:, f'M{mode_number} ' + 'Eigenfrequency (Hz)'].astype(float)

    for param in param_limits.keys():
        for mode_number in range(1, 5):

            # Consequentially limiting specified parameters
            df = catch_limit_overflow(df, param_name=f'M{mode_number} ' + param,
                                      lower_limit=param_limits[param][0],
                                      upper_limit=param_limits[param][1])
    return df
