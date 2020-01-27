import pandas as pd


def load_data(file_name, sep=";"):
    df = pd.read_csv(file_name, sep)

    df["y"] = df["y"].map(lambda x: 1 if str(x) == "yes" else 0).astype(float)

    for c in df.columns:
        if df[c].dtype == "object":
            labels = {value: index for index, value in enumerate(["__UNKNOWN__"] + sorted(list(set(df[c].astype(str).values))))}
            df[c] = df[c].map(lambda x: labels.get(x)).fillna(labels["__UNKNOWN__"]).astype(int)

    return df.drop(["y", "duration"], axis=1), df["y"]
