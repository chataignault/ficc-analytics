# %% [code]

# %% [code]
"""
The evaluation API requires that you set up a server which will respond to inference requests.
We have already defined the server; you just need write the predict function.
When we evaluate your submission on the hidden test set the client defined in `mitsui_gateway` will run in a different container
with direct access to the hidden test set and hand off the data timestep by timestep.

Your code will always have access to the published copies of the competition files.
"""

import os
import time
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

from kaggle_evaluation.core.base_gateway import GatewayRuntimeError
import kaggle_evaluation.mitsui_inference_server


NUM_TARGET_COLUMNS = 424

try:
    train = pl.read_csv("/kaggle/input/mitsui-commodity-prediction-challenge/train.csv")
    train_labels = pl.read_csv(
        "/kaggle/input/mitsui-commodity-prediction-challenge/train_labels.csv"
    )
except FileNotFoundError:
    print("Import train dataset from current directory")
    train = pl.read_csv("train.csv")
    train_labels = pl.read_csv("train_labels.csv")

print(train.shape, train_labels.shape)
print(train_labels.head())  # date_id index column

# train model
lin = Ridge()
param_distribution = {"alpha": np.logspace(-4, 0, num=4)}
print(param_distribution)
clf = RandomizedSearchCV(lin, param_distribution, random_state=0)

X = train.select(pl.exclude("date_id").forward_fill().backward_fill()).to_numpy()
Y = train_labels.select(pl.exclude("date_id").forward_fill().backward_fill()).to_numpy()


mu = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_std = (X - mu) / std

search = clf.fit(X_std, Y)
alpha = search.best_params_["alpha"]
print(X.shape, Y.shape)
print("Best regularisation parameter :", alpha)
lin = Ridge(alpha=alpha)
lin.fit(X_std, Y)


# train score
Y_hat = lin.predict(X_std)
mse = np.linalg.norm(Y - Y_hat) / len(Y)
r2 = 1 - np.var(Y - Y_hat) / np.var(Y)

print("Train MSE :", mse)
print("Train R2 :", r2)

def predict(
    test: pl.DataFrame,
    label_lags_1_batch: pl.DataFrame,
    label_lags_2_batch: pl.DataFrame,
    label_lags_3_batch: pl.DataFrame,
    label_lags_4_batch: pl.DataFrame,
) -> pl.DataFrame | pd.DataFrame:
    """Replace this function with your inference code.
    You can return either a Pandas or Polars dataframe, though Polars is recommended for performance.
    Each batch of predictions (except the very first) must be returned within 5 minutes of the batch features being provided.
    """
    if len(test) == 0:
        # default prediction
        predictions = pl.DataFrame(
            {f"target_{i}": i / 1000 for i in range(NUM_TARGET_COLUMNS)}
        )
    else:
        # predict with the linear regression
        x = test.fill_null(0.0).select(pl.exclude(["date_id", "is_scored"])).to_numpy()
        x[x == None] = 0.0
        x = x.astype(float)
        x = x - mu
        x = x / std
        pred = lin.predict(x)
        predictions = pl.DataFrame(
            {f"target_{i}": pred[0][i] for i in range(NUM_TARGET_COLUMNS)}
        )

    assert isinstance(predictions, (pd.DataFrame, pl.DataFrame))
    assert len(predictions) == 1
    return predictions


# When your notebook is run on the hidden test set, inference_server.serve must be called within 15 minutes of the notebook starting
# or the gateway will throw an error. If you need more than 15 minutes to load your model you can do so during the very
# first `predict` call, which does not have the usual 1 minute response deadline.
inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(
    predict
)

# server is the configured grpc server object.
# debug gRPC server 
for service in inference_server.server._state.generic_handlers:
    print("Service Name:", service.service_name())
    for method in service._method_handlers:
        print(4*" " + method)

# run the server gateway
if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    if os.path.exists("/kaggle"):
        try:
            inference_server.run_local_gateway(
                ("/kaggle/input/mitsui-commodity-prediction-challenge/",)
            )
        except GatewayRuntimeError as e:
            print(f"{e}")
    else:
        inference_server.run_local_gateway(
            (".",)
        )

# %% estimate local train score
try:
    submission = pd.read_parquet("submission.parquet")
    print(submission)
except FileNotFoundError:
    print("Submission file not found")
