import numpy as np
from sklearn.datasets import make_moons, fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder, StandardScaler
import torch
import torchvision
import torchvision.transforms as transforms
import math


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin. The angle should be given in radians.
    Args:
        origin ([float, float]): x and y coordinate of the origin around which to rotate;
        point ([float, float]): x and y coordinate of the point to rotate;
        angle (float): angle of rotation, in radians.
    return:
        [float, float], coordinates of the rotated point.
    """
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def gen(dataset, m, d, shuffle=True):
    """
    Generates a linearly separable 2-classes d-dimensional dataset. Each class
        is normally distributed around a random mean with identity covariance matrix.
    Args:
        dataset (str): Name of the dataset to generate;
        m (int): Number of examples in each class
        d (int): Input dimension of each dataset
        shuffle (bool): Whether the dataset is shuffled
    return:
        Tuple
    """
    x = []
    if dataset == 'both':
        rand = np.random.randint(2)  # Randomly chooses between 'easy' and 'moons' dataset
        dataset = rand * 'easy' + (1 - rand) * 'moons'
    if dataset == 'easy':  # Corresponds to Gaussian blobs
        mu = np.random.rand(d) * 10 - 5  # Random center for the Gaussian blobs
        x = np.random.multivariate_normal(mean=mu, cov=np.eye(d), size=2 * m)
        x[:m] += np.sign(np.random.rand(d) - 0.5) * 5  # Making a class distinct from the other
    if dataset == 'moons':
        assert d == 2  # Moons must be of dimension 2
        # Three parameters constitute a dataset: the origin of the moons, their rotation around the origin and its scale
        scale = np.random.randint(3, 8)
        rotation = np.random.randint(0, 360)
        origin = np.random.randint(-10, 10, 2)
        x = make_moons(n_samples=2 * m, shuffle=False, noise=0.08)[0] * scale + origin
        for i in range(2 * m):
            x[i, 0], x[i, 1] = rotate((0, 0), x[i], math.radians(rotation))  # We rotate, one by one, the points
    y = np.ones((2 * m, 1))
    y[:m] -= 1
    if shuffle:
        indx = np.arange(2 * m)  # Randomize the position of the points in the dataset
        np.random.shuffle(indx)
        x, y = x[indx], y[indx]
    return np.hstack((x, y * 2 - 1))


def data_gen(dataset, n, m, d, shuffle=True):
    """
    Generates a set of linearly separable 2-classes d-dimensional datasets.
    Args:
        dataset (str): Name of the dataset to generate;
        n (int): Number of linearly separable datasets to create;
        m (int): Number of examples per dataset;
        d (int): Input dimension of each dataset;
        shuffle (bool): whether to shuffle the points in each generated dataset
    return:
        Numpy array of dims n x m x d
    """
    assert d == 2  # Only generates 2D datasets
    m, data = int(m / 2), np.zeros((n, m, 3))
    for i in range(n):
        data[i] = gen(dataset, m, d, shuffle)
    return data


def load_mnist():
    """
    Generates a set of 90 MNIST binary sub-problems
    return:
        Numpy array of dims n x m x d
    """
    # Loading the initial dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./datasets', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./datasets', train=False, download=True, transform=transform)
    trainset = torch.hstack((trainset.data.reshape((60000, 28 * 28)), trainset.targets.reshape(60000, -1)))
    testset = torch.hstack((testset.data.reshape((10000, 28 * 28)), testset.targets.reshape(10000, -1)))
    dataset = torch.vstack((trainset, testset))

    # Creating 90 MNIST binary sub-problems
    data, k = np.zeros((90, 6313 * 2, 28 * 28)), 0
    indx = np.arange(2 * 6313)  # Randomize the position of the points in the dataset
    for i in range(10):
        for j in range(10):
            if i != j:
                x_1 = dataset[dataset[:, -1] == i, :-1]
                x_2 = dataset[dataset[:, -1] == j, :-1]
                x = torch.vstack((x_1[:6313], x_2[:6313]))
                y = torch.ones((6313 * 2, 1))
                y[:6313] -= 1
                np.random.shuffle(indx)
                data[k] = torch.hstack((x, y * 2 - 1))[indx]
                k += 1
    return data


def load_MTPL(task, n, m):
    """
    Fetch the French Motor Third-Party Liability Claims dataset.
    Args:
        task (str): the task to perform on the MTPL2 dataset (choices: 'frequency', 'severity', 'pure').
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
    for i in range(len(df['Region'].cat.categories)):
        if k == n:
            break
        inds = df['Region'] == df['Region'].cat.categories[i]
        for j in range(len(x[inds]) // m):
            new_data[k] = data[inds][int(j*m): int((j+1)*m)]
            k += 1
            if k == n:
                break
    return new_data[:k]
