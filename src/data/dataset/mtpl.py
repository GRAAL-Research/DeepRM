import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, KBinsDiscretizer, OneHotEncoder


def load_MTPL(task, n, m):
    """
    Fetch the French Motor Third-Party Liability Claims dataset.
    Args:
        task (str): the task to perform on the MTPL2 dataset (choices: "frequency", "severity", "pure").
        n (int): Number of linearly separable datasets to create;
        m (int): Number of examples per dataset.
    return:
        Numpy array of dims n x m x d
    """
    # freMTPL2freq dataset from https://www.openml.org/d/41214
    df_freq = fetch_openml(data_id=41214, as_frame=True).data
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)
    df_freq.set_index("IDpol", inplace=True)

    # freMTPL2sev dataset from https://www.openml.org/d/41215
    df_sev = fetch_openml(data_id=41215, as_frame=True).data

    # sum ClaimAmount over identical IDs
    df_sev = df_sev.groupby("IDpol").sum()

    df = df_freq.join(df_sev, how="left")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

    # unquote string fields
    for column_name in df.columns[df.dtypes.values == object]:
        df[column_name] = df[column_name].str.strip("'")

    # Correct for unreasonable observations (that might be data error)
    # and a few exceptionally large claim amounts
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1)
    df["ClaimAmount"] = df["ClaimAmount"].clip(upper=200000)

    # If the claim amount is 0, then we do not count it as a claim. The loss function
    # used by the severity model needs strictly positive claim amounts. This way
    # frequency and severity are more consistent with each other.
    df.loc[(df["ClaimAmount"] == 0) & (df["ClaimNb"] >= 1), "ClaimNb"] = 0

    log_scale_transformer = make_pipeline(
        FunctionTransformer(func=np.log), StandardScaler()
    )

    column_trans = ColumnTransformer(
        [
            (
                "binned_numeric",
                KBinsDiscretizer(n_bins=10, random_state=0),
                ["VehAge", "DrivAge"],
            ),
            (
                "onehot_categorical",
                OneHotEncoder(),
                ["VehBrand", "VehPower", "VehGas", "Area", "Region"],
            ),
            ("passthrough_numeric", "passthrough", ["BonusMalus"]),
            ("log_scaled_numeric", log_scale_transformer, ["Density"]),
        ],
        remainder="drop",
    )
    x, y, weight = column_trans.fit_transform(df).toarray(), [], []
    if task == "frequency":
        df["Frequency"] = df["ClaimNb"] / df["Exposure"]
        y = np.array(df["Frequency"].to_numpy() > 0, dtype=int).reshape((-1, 1))
        weight = df["Exposure"].to_numpy().reshape((-1, 1))
    elif task == "severity":
        df["AvgClaimAmount"] = df["ClaimAmount"] / np.fmax(df["ClaimNb"], 1)
        mask = df["ClaimAmount"] > 0
        x = x[mask.values]
        y = df["AvgClaimAmount"][mask.values].to_numpy().reshape((-1, 1))
        weight = df["ClaimNb"][mask.values].to_numpy().reshape((-1, 1))
    elif task == "pure":
        df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]
        y = df["PurePremium"].to_numpy().reshape((-1, 1))
        weight = df["Exposure"].to_numpy().reshape((-1, 1))
    data = np.hstack((np.array(x), weight, y * 2 - 1)).astype(float, copy=False)
    new_data, k = np.zeros((n, m, 77)), 0
    for i in range(len(df["Region"].cat.categories)):
        if k == n:
            break
        inds = df["Region"] == df["Region"].cat.categories[i]
        for j in range(len(x[inds]) // m):
            new_data[k] = data[inds][int(j * m): int((j + 1) * m)]
            k += 1
            if k == n:
                break
    return new_data[:k]
