import argparse
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.preprocess import load_data
from src.utility import seed_everything

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("n_folds", type=int)

    args = parser.parse_args()

    SEED = args.seed
    N_FOLDS = args.n_folds

    seed_everything(SEED)

    X, y = load_data("data/bank-additional-full.csv")
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    avg_score = 0

    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        print("Fold:", fold_idx, flush=True)

        X_train, y_train = X.iloc[train_idx, :], y[train_idx]
        X_valid, y_valid = X.iloc[valid_idx, :], y[valid_idx]

        model = LogisticRegression(
            tol=0.014562448890118148,
            C=9.256722875165577,
            fit_intercept=True,
            class_weight="balanced",
            solver="newton-cg",
            max_iter=120,
            warm_start=True,
            random_state=SEED
        )
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_valid).astype(float)[:, 1]
        score = roc_auc_score(list(y_valid), y_pred)
        avg_score += score

        print("logistic regression score:",  score, flush=True)

    avg_score /= N_FOLDS

    print("average score:", avg_score, flush=True)


if __name__ == "__main__":
    main()
