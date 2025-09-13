# %% [code]
# %% [code]
# # %% [bash]

# !pip install scikit-learn

# %% [code]
"""
The evaluation API requires that you set up a server which will respond to inference requests.
We have already defined the server; you just need write the predict function.
When we evaluate your submission on the hidden test set the client defined in `mitsui_gateway` will run in a different container
with direct access to the hidden test set and hand off the data timestep by timestep.

Your code will always have access to the published copies of the competition files.
"""

import os

import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.ensemble import RandomForestRegressor
import copy 

import kaggle_evaluation.mitsui_inference_server

print(os.listdir(os.path.join(os.getcwd(), "..", "input", "mitsui-commodity-prediction-challenge")))

NUM_TARGET_COLUMNS = 424

train = pl.read_csv("/kaggle/input/mitsui-commodity-prediction-challenge/train.csv")
train_labels = pl.read_csv("/kaggle/input/mitsui-commodity-prediction-challenge/train_labels.csv")

print(train.shape, train_labels.shape)
print(train_labels.head()) # date_id index column

# train model
lin = LinearRegression()
lin = Ridge(alpha=.1)

X = train.fill_null(0.).select(pl.exclude("date_id")).to_numpy()
Y = train_labels.fill_null(0.).select(pl.exclude("date_id")).to_numpy()
print(X.shape, Y.shape)
lin.fit(X, Y)


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
        predictions = pl.DataFrame({f'target_{i}': i / 1000 for i in range(NUM_TARGET_COLUMNS)})
    else:
        # predict with the linear regression
        x = test.fill_null(0.).select(pl.exclude(["date_id", "is_scored"])).to_numpy()
        x[x == None] = 0.
        x = x.astype(float)
        pred = lin.predict(x)
        predictions = pl.DataFrame({f'target_{i}': pred[0][i] for i in range(NUM_TARGET_COLUMNS)})

    assert isinstance(predictions, (pd.DataFrame, pl.DataFrame))
    assert len(predictions) == 1
    return predictions


# When your notebook is run on the hidden test set, inference_server.serve must be called within 15 minutes of the notebook starting
# or the gateway will throw an error. If you need more than 15 minutes to load your model you can do so during the very
# first `predict` call, which does not have the usual 1 minute response deadline.
inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/mitsui-commodity-prediction-challenge/',))
