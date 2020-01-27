import argparse
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
from src.preprocess import load_data
from src.utility import seed_everything

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("n_folds", type=int)
    parser.add_argument("output_features", type=bool, default=False)

    args = parser.parse_args()

    SEED = args.seed
    N_FOLDS = args.n_folds

    seed_everything(SEED)

    X, y = load_data("data/bank-additional-full.csv")
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    avg_score = 0

    feature_importance = pd.DataFrame()
    feature_importance["Feature"] = X.columns
    feature_importance["Value"] = 0

    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        print("Fold:", fold_idx, flush=True)

        X_train, y_train = X.iloc[train_idx, :], y[train_idx]
        X_valid, y_valid = X.iloc[valid_idx, :], y[valid_idx]

        model = xgb.XGBRegressor(
            n_estimators=486,
            max_depth=23,
            learning_rate=0.014315933846251667,
            booster="gbtree",
            tree_method="exact",
            gamma=0.7581225878358416,
            subsample=0.9340339327920703,
            colsample_bytree=0.6940772015224637,
            colsample_bylevel=0.559247335020885,
            colsample_bynode=0.7962006061767392,
            reg_alpha=0.6394227535273009,
            reg_lambda=0.19510772446939947,
            scale_pos_weight=0.8349805523658489,
            objective="reg:squarederror",
            random_state=SEED
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_valid).astype(float)
        score = roc_auc_score(list(y_valid), y_pred)
        avg_score += score

        print("xgboost score:", score, flush=True)

        current_importance = pd.DataFrame(zip(X.columns, model.feature_importances_), columns=["Feature", "Value"])
        feature_importance = pd.concat((feature_importance, current_importance)).groupby("Feature", as_index=False).sum()

    avg_score /= N_FOLDS

    print("average score:", avg_score, flush=True)

    if args.output_features:
        feature_importance["Value"] *= 100 / feature_importance["Value"].sum()

        fig = plt.figure(figsize=(20, 20))
        fig.patch.set_facecolor("white")
        sns.set(style="whitegrid")
        sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))
        plt.title("Feature importance (%)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
