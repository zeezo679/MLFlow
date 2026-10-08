import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import mlflow
import mlflow.sklearn

N_ESTIMATORS = 100  # kept identical across runs so only the two hyperparameters change

RUNS = [
    {"name": "run_1", "max_depth": 3, "learning_rate": 0.1},
    {"name": "run_2", "max_depth": 5, "learning_rate": 0.05},
    {"name": "run_3", "max_depth": 7, "learning_rate": 0.01},
]

TEST_SIZE = 0.2 #20% VALIDATION
SEED = 42

EXPERIMENT_NAME = "california-housing-gbr"

def load_data():
    X, y = fetch_california_housing(as_frame=True, return_X_y=True)
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

def compute_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def train_model(X_train, y_train, max_depth, learning_rate):
    model = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=SEED,
    )
    model.fit(X_train, y_train)
    return model

def train_and_log(cfg, X_train, X_val, y_train, y_val):
    with mlflow.start_run(run_name=cfg["name"]) as run:
        mlflow.log_params({
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
            "n_estimators": N_ESTIMATORS,
            "seed": SEED
        })

        model = train_model(X_train, y_train, cfg["max_depth"], cfg["learning_rate"])
        metrics = compute_metrics(y_val, model.predict(X_val))

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])

    return {"run_id": run.info.run_id, "name": cfg["name"], **cfg, **metrics}

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_val, y_train, y_val = split_data(X, y)

    mlflow.set_experiment(EXPERIMENT_NAME)
    results = [train_and_log(cfg, X_train, X_val, y_train, y_val) for cfg in RUNS]

    for r in results:
        print(r["name"], r["max_depth"], r["learning_rate"], round(r["rmse"], 4))



