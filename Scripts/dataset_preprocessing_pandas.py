import pandas as pd


def split_df(df, num_common):
    """
    Splits full dataframe, containing both X and Y data, into two separate DataFrames of X and Y parts.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be split into X and Y parts.
        This dataframe should contain correct column names.
    num_common : int
        Number of parameters which are general for all resonant modes.
        They should go first in df.columns.

    Returns
    ----------
    (df_x, df_y): (pd.DataFrame, pd.DataFrame)
        Tuple containing X and Y dataframes
    """
    all_params = list(df.columns)
    params_x = all_params[0:num_common]
    params_y = all_params[num_common:]

    return df[params_x], df[params_y]


class Scaler:
    def __init__(self, scaler_type='standard', mode_number=4):
        """
        Parameters
        ----------
        scaler_type : str
            Type of preferred scaler. 'Standard' and 'Robust' are available (default 'Standard').
        mode_number : int
            Number of resonant modes in the dataset (default 4).

        Returns
        ----------
        None
        """
        self.mode_number = mode_number
        self.cols_x = None
        self.cols_y = None
        self.log_params = None

        if scaler_type == 'standard':
            self.s_x = StandardScaler()
            self.s_y = StandardScaler()
        elif scaler_type == 'robust':
            self.s_x = RobustScaler()
            self.s_y = RobustScaler()

    def fit(self, x_train, y_train, log_params=None):
        """
        Fits scaler with the training data.

        Parameters
        ----------
        x_train : pd.DataFrame
            Training X data.
        y_train : pd.DataFrame
            Training Y data.
        log_params : list of str
            List containing parameter names which require log10 to be taken before applying them to scaler.

        Returns
        ----------
        None
        """
        self.cols_x = x_train.columns
        self.cols_y = y_train.columns
        self.log_params = log_params

        # Making copy of x_train and y_train.
        # Otherwise log10 operation will be applied twice as x_train and y_train are transformed in the next pipeline step.
        x_train_copy, y_train_copy = x_train.copy(), y_train.copy()

        # These parameters require log10 operation in my dataset
        x_train_copy.loc[:, 'Beam length (um)'] = np.log10(x_train_copy.loc[:, 'Beam length (um)'])
        x_train_copy.loc[:, 'Temperature (K)'] = np.log10(x_train_copy.loc[:, 'Temperature (K)'])

        # Setting the default parameters for log10 operation
        if self.log_params is None:
            self.log_params = ['Eigenfrequency (Hz)', 'Quality factor',
                               'Effective mass (kg)', 'TED (W)',
                               'Noise (kg^2/s^3)']

        # Taking log10 operation of log_params
        for mode_number in range(1, self.mode_number + 1):
            for param in self.log_params:
                y_train_copy.loc[:, f'M{mode_number} ' + param] = np.log10(y_train_copy.loc[:, f'M{mode_number} ' + param])

        # Fitting log10-ed params in scaler
        self.s_x.fit(x_train_copy)
        self.s_y.fit(y_train_copy)

    def transform(self, x, y):
        """
        Transforms data.

        Parameters
        ----------
        x : pd.DataFrame
            X-data to be transformed.
        y : pd.DataFrame
            Y-data to be transformed.

        Returns
        ----------
        x : pd.DataFrame
            Transformed X-data.
        y : pd.DataFrame
            Transformed Y-data.
        """
        x.loc[:, 'Beam length (um)'] = np.log10(x.loc[:, 'Beam length (um)'])
        x.loc[:, 'Temperature (K)'] = np.log10(x.loc[:, 'Temperature (K)'])

        for mode_number in range(1, self.mode_number + 1):
            for param in self.log_params:
                y.loc[:, f'M{mode_number} ' + param] = np.log10(y.loc[:, f'M{mode_number} ' + param])

        # Scaling data
        x = pd.DataFrame(self.s_x.transform(x))
        x.columns = self.cols_x

        y = pd.DataFrame(self.s_y.transform(y))
        y.columns = self.cols_y

        return x, y

    def reverse_transform(self, x, y, concat_required: bool):
        """
        Performs reverse transform on data. Can be used in order to get physical numbers from model outputs.

        Parameters
        ----------
        x : pd.DataFrame
            Transformed X-data.
        y : pd.DataFrame
            Transformed Y-data (e.g. model outputs).
        concat_required: bool
            If True returns reverse-transformed data in a single DataFrame (concatenated output).

        Returns
        ----------
        x : pd.DataFrame
            Reverse-transformed X-data.
        y : pd.DataFrame
            Reverse-transformed X-data.
        """
        x = pd.DataFrame(self.s_x.inverse_transform(x))
        x.columns = self.cols_x

        y = pd.DataFrame(self.s_y.inverse_transform(y))
        y.columns = self.cols_y

        # Taking power of values
        x.loc[:, 'Beam length (um)'] = np.power(10, x.loc[:, 'Beam length (um)'])
        x.loc[:, 'Temperature (K)'] = np.power(10, x.loc[:, 'Temperature (K)'])

        for mode_number in range(1, self.mode_number + 1):
            for param in self.log_params:
                y.loc[:, f'M{mode_number} ' + param] = np.power(10, y.loc[:, f'M{mode_number} ' + param])

        if concat_required:
            return pd.concat([x, y], axis=1)
        else:
            return x, y

    def transform_real_x(self, x):
        """
        Performs (forward) transform on X-data only.

        Parameters
        ----------
        x : pd.DataFrame
            X-data to be transformed.

        Returns
        ----------
        x : pd.DataFrame
            Transformed X-data.
        """
        x.loc[:, 'Beam length (um)'] = np.log10(x.loc[:, 'Beam length (um)'])
        x.loc[:, 'Temperature (K)'] = np.log10(x.loc[:, 'Temperature (K)'])

        # Scaling data
        x = pd.DataFrame(self.s_x.transform(x))
        x.columns = self.cols_x

        return x

    def transform_real_y(self, y):
        """
        Performs (forward) transform on Y-data only.

        Parameters
        ----------
        y : pd.DataFrame
            Y-data to be transformed.

        Returns
        ----------
        y : pd.DataFrame
            Transformed Y-data.
        """
        for mode_number in range(1, 5):
            for param in self.log_params:
                y.loc[:, f'M{mode_number} ' + param] = np.log10(y.loc[:, f'M{mode_number} ' + param])

        # Scaling data
        y = pd.DataFrame(self.s_y.transform(y))
        y.columns = self.cols_y

        return y
