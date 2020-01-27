import warnings

from hyperopt import STATUS_OK, fmin, hp, tpe
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
from src.preprocess import load_data
from src.utility import seed_everything

warnings.filterwarnings("ignore")

SEED = 123
N_FOLDS = 5

seed_everything(SEED)

X, y = load_data("data/bank-additional-full.csv")

folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def objective(args):
    avg_score = 0

    for train_idx, valid_idx in folds.split(X, y):
        X_train, y_train = X.iloc[train_idx, :], y[train_idx]
        X_valid, y_valid = X.iloc[valid_idx, :], y[valid_idx]

        model = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=-1, random_state=SEED, **args)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_valid).astype(float)
        avg_score += roc_auc_score(list(y_valid), y_pred)

    return {"loss": 1 - avg_score / N_FOLDS, "status": STATUS_OK}


def main():
    search_space = {
        "n_estimators": hp.choice("n_estimators", range(1, 500)),
        "max_depth": hp.choice("max_depth", range(1, 50)),
        "learning_rate": hp.loguniform("learning_rate", -10, 0),
        "booster": hp.choice("booster", ["gbtree", "dart"]),
        "tree_method": hp.choice("tree_method", ["exact", "approx", "hist"]),
        "gamma": hp.uniform("gamma", 0, 1),
        "subsample": hp.uniform("subsample", 0, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0, 1),
        "colsample_bylevel": hp.uniform("colsample_bylevel", 0, 1),
        "colsample_bynode": hp.uniform("colsample_bynode", 0, 1),
        "reg_alpha": hp.uniform("reg_alpha", 0, 1),
        "reg_lambda": hp.uniform("reg_lambda", 0, 1),
        "scale_pos_weight": hp.uniform("scale_pos_weight", 0, 1)
    }

    best = fmin(objective, space=search_space, algo=tpe.suggest, max_evals=100)

    print(best)

    with open("xgboost_best_params.out", "w") as f:
        f.write(str(best))


if __name__ == "__main__":
    main()
