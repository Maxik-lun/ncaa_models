import pandas as pd
import numpy as np
import shap
from sklearn.metrics import mean_absolute_error

def season_validation(model, X, y, season_groups):
    seasons = [
        2010, 2011, 2012, 2013, 2014, 2015, 2016, 
        2017, 2018, 2019, 2021, 2022, 2023, 2024]
    y_pred = pd.Series(np.nan, index=y.index)
    models = []
    for season in seasons:
        mask = season_groups == season
        X_train = X[~mask]
        y_train = y[~mask]
        X_val = X[mask]
        y_val = y[mask]
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)])
        models.append(model)
        y_pred[mask] = model.predict(X_val)
        print(season, mean_absolute_error(y_val, y_pred[mask]))
    return y_pred, models

