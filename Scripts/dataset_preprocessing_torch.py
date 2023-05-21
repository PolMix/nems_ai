import pandas as pd
import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, cols_to_save=None):
        """
        Creates new dataset class object.

        Parameters
        ----------
        x : pd.DataFrame
            Dataframe that contains X-data
        y : pd.DataFrame
            Dataframe that contains Y-data
        cols_to_save : str or None
            Column indices to be used in dataset. If None, all initial columns are used (default None).
        """
        if cols_to_save == None:
            self.x = torch.tensor(x.iloc[:, :].values, dtype=torch.float32)
            self.y = torch.tensor(y.iloc[:, :].values, dtype=torch.float32)
        else:
            self.x = torch.tensor(x.iloc[:, :].values, dtype=torch.float32)
            self.y = torch.tensor(y.iloc[:, cols_to_save].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]