import pandas as pd

def describe_clusters(X, labels, include_stats=('mean', 'std', 'min', '25%', '50%', '75%', 'max')):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    X_desc = X.copy()
    X_desc["cluster"] = labels

    num_cols = X_desc.select_dtypes(include=[np.number]).columns.drop("cluster", errors="ignore")

    grouped_desc = X_desc.groupby("cluster")[num_cols].describe().T

    df_desc = grouped_desc.loc[grouped_desc.index.get_level_values(1).isin(include_stats)]
    df_desc.index = [f"{var}_{stat}" for var, stat in df_desc.index]
    df_desc = df_desc.round(3).sort_index()

    return df_desc