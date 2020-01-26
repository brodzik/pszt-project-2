import os
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

warnings.filterwarnings("ignore")

SEED = 123
N_FOLDS = 5


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def main():
    df = pd.read_csv("data/bank-additional-full.csv", sep=";")

    df["y"] = df["y"].map(lambda x: 1 if str(x) == "yes" else 0).astype(float)

    for c in df.columns:
        if df[c].dtype == "object":
            labels = {value: index for index, value in enumerate(["__UNKNOWN__"] + sorted(list(set(df[c].astype(str).values))))}
            df[c] = df[c].map(lambda x: labels.get(x)).fillna(labels["__UNKNOWN__"]).astype(int)

    X, y = df.drop(["y", "duration"], axis=1), df["y"]

    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        print("Fold:", fold_idx + 1, flush=True)

        X_train, y_train = X.iloc[train_idx, :], y[train_idx]
        X_valid, y_valid = X.iloc[valid_idx, :], y[valid_idx]

        model = LogisticRegression(solver="lbfgs", random_state=SEED)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_valid).astype(float)[:, 1]
        print("logistic regression score:", roc_auc_score(list(y_valid), y_pred), flush=True)

        model = xgb.XGBRegressor(objective="reg:squarederror", random_state=SEED)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid).astype(float)
        print("XGBoost score:", roc_auc_score(list(y_valid), y_pred), flush=True)


if __name__ == "__main__":
    seed_everything(SEED)
    main()
