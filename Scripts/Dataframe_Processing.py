{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO5RS0I3j61i3Ol8/4M9peV",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/PolMix/nems_ai/blob/main/Scripts/Dataframe_Processing.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bYFnRSr4-OXB"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "\n",
        "def split_df(df, general_param_number, mode_number):\n",
        "    \"\"\"\n",
        "    Splits full dataframe, containing both X and Y data, into two separate DataFrames of X and Y parts.\n",
        "\n",
        "    Parameters\n",
        "    ----------\n",
        "    df : pd.DataFrame\n",
        "        Dataframe to be split into X and Y parts.\n",
        "        This dataframe should contain correct column names.\n",
        "    general_param_number : int\n",
        "        Number of parameters which are general for all resonant modes.\n",
        "        They should go first in df.columns.\n",
        "    mode_number : int\n",
        "        Number of modes in dataset.\n",
        "\n",
        "    Returns\n",
        "    ----------\n",
        "    (df_x, df_y): (pd.DataFrame, pd.DataFrame)\n",
        "        Tuple containing X and Y dataframes\n",
        "    \"\"\"\n",
        "    all_params = list(df.columns)\n",
        "    params_x = all_params[0:general_param_number]\n",
        "    params_y = all_params[general_param_number:]\n",
        "\n",
        "    return df[params_x], df[params_y]"
      ]
    }
  ]
}