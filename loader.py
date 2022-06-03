import os
import pandas as pd


def load_data(folder=''):
    dfs = []
    conditions = ["exp0", "exp1", "exp2"]
    for condition in conditions:
        df = pd.read_pickle(os.path.join(folder, f"annotations_{condition}.pkl"))
        dfs.append(df)
    return dfs


def load_stimuli(folder=''):
    dfs = []
    conditions = ["123", "456"]
    for condition in conditions:
        df = pd.read_pickle(os.path.join(folder, f"stimuli{condition}.pkl"))
        dfs.append(df)
    return dfs
