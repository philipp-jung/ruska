import json
import logging
import urllib.parse
import random
import datetime
import requests
import numpy as np
import pandas as pd
from typing import List, Union, Callable
from pathlib import Path


def reduce_runs_list(
    result,
    config_of_interest=["dataset", "error_fraction"],
    metrics=["f1", "precision", "recall"],
    run_label="run",
):
    """
    Same as reduce_runs, but when the performance_label doesn't yield a single float, but a
    list of floats instead.
    """
    labels_of_interest = config_of_interest + [run_label] + metrics
    result_enc = [
        {**r["config"], **{m: json.dumps(r["result"][m]) for m in metrics}}
        for r in result
    ]
    df = pd.DataFrame(
        result_enc
    )  # use pandas' groupby implementation to make life easier
    groups_label = [x for x in labels_of_interest if x not in metrics + [run_label]]
    df_grouped = df.loc[:, labels_of_interest].groupby(groups_label).agg(list)
    grouped_dict = df_grouped.reset_index().to_dict("records")

    new_result = []

    for r in grouped_dict:
        d = {key: r[key] for key in groups_label}
        d["n_runs"] = len(r[run_label])
        for m in metrics:
            metric_values = np.array([json.loads(f) for f in r[m]])
            d[f"{m}_avg"] = np.ndarray.mean(metric_values, axis=0)
        new_result.append(d)
    return new_result


def reduce_runs(
    result, performance_labels=["precision", "recall", "f1"], run_label="run"
):
    """
    The way I run experiments, results are usually a list of dictionaries. These dictionaries all have
    the same keys, but different values. The keys generally vary with the experiment and contain parameter
    names, while the values contain the parameter.

    One parameter that is recurring is 'run'. Since I have to work with multiple runs to compute averages,
    I frequently average dictionaries with the same parameters over all runs and return the average and
    standard error.

    This function does just that -- when a list of dictionaries has the size sum(n_parameters)*n_runs,
    this reduces the list to sum(n_parameters), and adds to each dictionary the keys 'average' and
    'standard_error'.
    """
    df = pd.DataFrame(result)  # use pandas' groupby implementation to make life easier
    groups_label = [
        x for x in result[0].keys() if x not in performance_labels + [run_label]
    ]
    df.loc[:, groups_label] = df.loc[:, groups_label].astype(str)
    df_subset = df.loc[:, groups_label + performance_labels]
    grouped = df_subset.groupby(groups_label).agg(list)
    grouped_dict = grouped.reset_index().to_dict("records")

    new_result = []

    # calculate se and avg for each metric
    for r in grouped_dict:
        d = {key: r[key] for key in groups_label}
        d["n_runs"] = len(r[performance_labels[0]])
        for metric in performance_labels:
            d[metric + "_avg"] = np.average(r[metric])
            d[metric + "_se"] = np.std(r[metric], ddof=1) / np.sqrt(np.size(r[metric]))
        new_result.append(d)

    return new_result


def get_distinct_list(l: list):
    result = []
    for x in l:
        if x not in result:
            result.append(x)
    return result


def format_delta(delta: datetime.timedelta):
    """Transforms timedelta to a human-readable format."""
    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    h = f"{int(hours)}h " if hours > 0 else ""
    m = f"{int(minutes)}m " if minutes > 0 else ""
    s = f"{int(seconds)}s"

    return h + m + s


def simple_mcar(df: pd.DataFrame, fraction: float, error_token=None):
    """
    Randomly insert missing values into a dataframe. Note that specifying the
    error_token as None preserves dtypes in the dataframe. If the error token
    is a string or a number, make sure to cast the entire dataframe to a dtype
    supporting categorical data with `df.astype(str)`. You will run into errors
    otherwise.

    Copies df, so that the clean dataframe you pass doesn't get corrupted
    in place.

    Note that casting to categorical data does mess up the imputer feature
    generator.
    """
    df_dirty = df.copy()
    n_rows, n_cols = df.shape

    if fraction > 1:
        raise ValueError("Cannot turn more than 100% of the values into errors.")
    target_corruptions = round(n_rows * n_cols * fraction)
    error_cells = random.sample(
        [(x, y) for x in range(n_rows) for y in range(n_cols)],
        k=target_corruptions,
    )

    for x, y in error_cells:
        df_dirty.iat[x, y] = error_token

    return df_dirty


def simple_mcar_column(se: pd.Series, fraction: float, error_token=None):
    """
    Randomly insert missing values into a pandas Series. See docs on
    simple_mcar for more information.

    Does not copy the passed Series `se`, so the Series you pass gets corrupted
    in-place and a variable assigned to it is returned.
    """
    n_rows = se.shape[0]
    target_corruptions = round(n_rows * fraction)
    error_positions = random.sample([x for x in range(n_rows)], k=target_corruptions)
    for x in error_positions:
        se.iat[x] = error_token
    return se


def send_notification(message: str, chat_id: Union[None, str], token: Union[None, str]):
    """
    Send a notification using a telegram bot called @ruska_experiment_bot.
    Secrets for this are stored locally in a .env file.
    """
    if chat_id is None or token is None:
        return True
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={urllib.parse.quote(message)}"
    _ = requests.get(url, timeout=10)
    return True


def wrap_experiment(experiment: Callable):
    def wrapped(i: int, logging_path: str, config: dict):
        logger = logging.getLogger(f"worker_{i}")
        fh = logging.FileHandler(logging_path, mode="a")
        # logger.addHandler(fh)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # logger.addHandler(ch)
        logging.basicConfig(level=logging.DEBUG, handlers=[fh, ch])
        logger.info(f"Starting experiment {i} with following config: {config}")
        experiment(config)
        logger.info(f"Experiment {i} finished.")

    return wrapped
