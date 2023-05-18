import pandas as pd


def split_df(df, general_param_number):
    """
    Splits full dataframe, containing both X and Y data, into two separate DataFrames of X and Y parts.
    
    Note that general parameters should go first in df.

    Parameters
    ----------
    - df : pd.DataFrame
        Dataframe to be split into X and Y parts.
    - general_param_number : int
        Number of parameters which are general for all resonant modes.

    Returns
    ----------
    (df_x, df_y): (pd.DataFrame, pd.DataFrame)
        Tuple containing X and Y dataframes
    """
    all_params = list(df.columns)
    params_x = all_params[0:general_param_number]
    params_y = all_params[general_param_number:]

    return df[params_x], df[params_y]
