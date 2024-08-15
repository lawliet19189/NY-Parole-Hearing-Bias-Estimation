import pickle
import pathlib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def save_model(model, imputer, scalar, used_features, dir: str = "models"):
    path = pathlib.Path(dir)
    path.mkdir(parents=True, exist_ok=True)
    full_path = path / "logistic_regression_model.pkl"
    with open(full_path, "wb") as f:
        pickle.dump(model, f)

    weights = model.weights
    intercept = model.bias
    full_path = path / "model_weights.npz"
    np.savez(full_path, weights=weights, intercept=intercept, used_features=used_features)

    with open(path / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)

    with open(path / "scalar.pkl", "wb") as f:
        pickle.dump(scalar, f)


def load_model(dir: str = "models"):
    path = pathlib.Path(dir)
    full_path = path / "logistic_regression_model.pkl"
    with open(full_path, "rb") as f:
        model = pickle.load(f)

    full_path = path / "model_weights.npz"
    data = np.load(full_path, allow_pickle=True)
    model.weights = data["weights"]
    model.bias = data["intercept"]
    used_features = data["used_features"]

    with open(path / "imputer.pkl", "rb") as f:
        imputer = pickle.load(f)

    with open(path / "scalar.pkl", "rb") as f:
        scalar = pickle.load(f)

    return model, imputer, scalar, used_features
