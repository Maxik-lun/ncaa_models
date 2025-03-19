import pandas as pd
from . import constants as ct

def get_tabular_dict(local_names):
    dfs = {}
    for c in local_names:
        key = '_'.join(c.split('_')[1:])
        try:
            dfs[key] = pd.read_csv(getattr(ct, c))
        except UnicodeDecodeError:
            print("Caught unicode error for", getattr(ct, c))
            dfs[key] = pd.read_csv(getattr(ct, c), encoding='unicode_escape')
    return dfs
