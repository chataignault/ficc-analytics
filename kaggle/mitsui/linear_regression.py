# %% [code]
# %% [code]
# %% [code]

import os
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from kaggle_evaluation.core.base_gateway import GatewayRuntimeError
import kaggle_evaluation.mitsui_inference_server


# %%
SOLUTION_NULL_FILLER = -999999


def rank_correlation_sharpe_ratio(merged_df: pd.DataFrame) -> float:
    """
    Calculates the rank correlation between predictions and target values,
    and returns its Sharpe ratio (mean / standard deviation).

    :param merged_df: DataFrame containing prediction columns (starting with 'prediction_')
                      and target columns (starting with 'target_')
    :return: Sharpe ratio of the rank correlation
    :raises ZeroDivisionError: If the standard deviation is zero
    """
    prediction_cols = [col for col in merged_df.columns if col.startswith('prediction_')]
    target_cols = [col for col in merged_df.columns if col.startswith('target_')]

    def _compute_rank_correlation(row):
        non_null_targets = [col for col in target_cols if not pd.isnull(row[col])]
        matching_predictions = [col for col in prediction_cols if col.replace('prediction', 'target') in non_null_targets]
        if not non_null_targets:
            raise ValueError('No non-null target values found')
        if row[non_null_targets].std(ddof=0) == 0 or row[matching_predictions].std(ddof=0) == 0:
            raise ZeroDivisionError('Denominator is zero, unable to compute rank correlation.')
        return np.corrcoef(row[matching_predictions].rank(method='average'), row[non_null_targets].rank(method='average'))[0, 1]

    daily_rank_corrs = merged_df.apply(_compute_rank_correlation, axis=1)
    std_dev = daily_rank_corrs.std(ddof=0)
    if std_dev == 0:
        raise ZeroDivisionError('Denominator is zero, unable to compute Sharpe ratio.')
    sharpe_ratio = daily_rank_corrs.mean() / std_dev
    return float(sharpe_ratio)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculates the rank correlation between predictions and target values,
    and returns its Sharpe ratio (mean / standard deviation).
    """
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    assert all(solution.columns == submission.columns)

    submission = submission.rename(columns={col: col.replace('target_', 'prediction_') for col in submission.columns})

    # Not all securities trade on all dates, but solution files cannot contain nulls.
    # The filler value allows us to handle trading halts, holidays, & delistings.
    solution = solution.replace(SOLUTION_NULL_FILLER, None)
    return rank_correlation_sharpe_ratio(pd.concat([solution, submission], axis='columns'))

# %% [code]
"""
The evaluation API requires that you set up a server which will respond to inference requests.
We have already defined the server; you just need write the predict function.
When we evaluate your submission on the hidden test set the client defined in `mitsui_gateway` will run in a different container
with direct access to the hidden test set and hand off the data timestep by timestep.

Your code will always have access to the published copies of the competition files.
"""


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
lin = LinearRegression()
# param_distribution = {"alpha": np.logspace(-4, 0, num=4)}
# print(param_distribution)
# clf = RandomizedSearchCV(lin, param_distribution, random_state=0)


train_processed = train.select(pl.exclude("date_id").forward_fill().backward_fill())
train_labels_processed = train_labels.select(pl.exclude("date_id").forward_fill().backward_fill())

X = train_processed.to_numpy()
Y = train_labels_processed.to_numpy()


mu = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_std = (X - mu) / std

# %% add the lagged targets to the dataset , and zeros where note available
Y_lag_1 = np.concatenate(
    [
        np.zeros((1, NUM_TARGET_COLUMNS)),
        Y[:-1]
    ]
)
print(Y_lag_1.shape, X_std.shape)
print(Y_lag_1[:5, :5])

X_std = np.concatenate(
    [
        X_std,
        Y_lag_1
    ],
    axis=1
)
print(X_std.shape)

# %%

# search = clf.fit(X_std, Y)
# alpha = search.best_params_["alpha"]
# print(X.shape, Y.shape)
# print("Best regularisation parameter :", alpha)
# lin = Ridge(alpha=alpha)
lin.fit(X_std, Y)
# Y_res = Y - lin.predict(X_std)

# tr = RandomForestRegressor(
#     criterion="friedman_mse",
#     # learning_rate=.1,
#     n_estimators=60
# )
# tr.fit(X_std, Y_res)

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
    test_lags = pl.concat(
        [
            label_lags_1_batch.drop(["date_id", "label_date_id"]),
            label_lags_2_batch.drop(["date_id", "label_date_id"]),
            label_lags_3_batch.drop(["date_id", "label_date_id"]),
            label_lags_4_batch.drop(["date_id", "label_date_id"]),
        ],
        how="horizontal"
    )
    if len(test) == 0:
        # default prediction
        predictions = pl.DataFrame(
            {f"target_{i}": i / 1000 for i in range(NUM_TARGET_COLUMNS)}
        )
    else:
        # predict with the linear regression
        x = test.fill_null(0.0).select(pl.exclude(["date_id", "is_scored"])).to_numpy()
        if len(test_lags):
            y_lagged = test_lags.fill_null(0.).select(pl.exclude(["date_id", "is_scored"])).to_numpy()
        else:
            y_lagged = np.zeros((1, test_lags.shape[1]))
        x[x == None] = np.array([mu])[x == None]
        x = x.astype(float)
        x = x - mu
        x = x / std
        x = np.concatenate(
            [
                x,
                y_lagged
            ],
            axis=1
        )
        pred = lin.predict(x) #+ tr.predict(x)
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
    submission = pd.read_parquet("/output/submission.parquet")
finally:
    # train score
    Y_hat = lin.predict(X_std)
    mse = np.linalg.norm(Y - Y_hat) / len(Y)
    r2 = 1 - np.var(Y - Y_hat) / np.var(Y)
    
    spearman_sharpe = score(train_labels.to_pandas(), pd.read_parquet("submission.parquet"), "date_id")
    
    print("Train MSE :", mse)
    print("Train R2 :", r2)
    print("Train Spearman Sharpe :", spearman_sharpe)
    

# %%
