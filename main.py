import pandas as pd
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


SEED = 123


def main():
    df = pd.read_csv("data/bank-additional-full.csv", sep=";")

    df["y"] = df["y"].map(lambda x: 1 if str(x) == "yes" else 0).astype(float)

    for c in df.columns:
        if df[c].dtype == "object":
            labels = {value: index for index, value in enumerate(["__UNKNOWN__"] + list(set(df[c].astype(str).values)))}
            df[c] = df[c].map(lambda x: labels.get(x)).fillna(labels["__UNKNOWN__"]).astype(int)

    X, y = df[:].drop("y", axis=1), df["y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    model = LogisticRegression(random_state=SEED)
    model.fit(X_train, y_train)

    print(model.score(X_test, y_test))

    y_pred = model.predict_proba(X_test).astype(float)[:, 1]
    print(roc_auc_score(list(y_test), y_pred))


if __name__ == "__main__":
    seed_everything(SEED)
    main()
