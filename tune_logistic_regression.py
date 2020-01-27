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

        model = LogisticRegression(n_jobs=-1, random_state=SEED, **args)
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_valid).astype(float)[:, 1]
        avg_score += roc_auc_score(list(y_valid), y_pred)

    return {"loss": 1 - avg_score / N_FOLDS, "status": STATUS_OK}


def main():
    search_space = {
        "tol": hp.lognormal("tol", 0, 1),
        "C": hp.lognormal("C", 0, 1),
        "fit_intercept": hp.choice("fit_intercept", [True, False]),
        "class_weight": hp.choice("class_weight", ["balanced", None]),
        "solver": hp.choice("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
        "max_iter": hp.choice("max_iter", range(10, 1000)),
        "warm_start": hp.choice("warm_start", [True, False])
    }

    best = fmin(objective, space=search_space, algo=tpe.suggest, max_evals=100)

    print(best)

    with open("logistic_regression_best_params.out", "w") as f:
        f.write(str(best))


if __name__ == "__main__":
    main()
