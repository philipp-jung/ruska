import json
import datetime
import numpy as np
import pandas as pd
from typing import List


def reduce_runs_list(
    result,
    config_of_interest=['dataset', 'error_fraction'],
    metrics = ['f1', 'precision', 'recall'],
    run_label="run"
):
    """
    Same as reduce_runs, but when the performance_label doesn't yield a single float, but a
    list of floats instead.
    """
    labels_of_interest = config_of_interest + [run_label] + metrics
    result_enc = [{**r['config'], **{m: json.dumps(r['result'][m]) for m in metrics}} for r in result]
    df = pd.DataFrame(result_enc)  # use pandas' groupby implementation to make life easier
    groups_label = [x for x in labels_of_interest if x not in metrics + [run_label]]
    df_grouped = df.loc[:, labels_of_interest].groupby(groups_label).agg(list)
    grouped_dict = df_grouped.reset_index().to_dict("records")

    new_result = []

    for r in grouped_dict:
        d = {key: r[key] for key in groups_label}
        d['n_runs'] = len(r[run_label])
        for m in metrics:
            metric_values = np.array([json.loads(f) for f in r[m]])
            d[f'{m}_avg'] = np.ndarray.mean(metric_values, axis=0)
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


def estimate_time_to_finish(times: List[datetime.datetime], total_runs: int):
    deltas = []
    i = 1

    while i < len(times):
        deltas.append(times[i] - times[i - 1])
        i += 1

    avg = sum(deltas, datetime.timedelta()) / len(deltas)
    current_run = len(times) - 1
    fd = format_delta
    eta = avg * (total_runs - current_run)
    return f"Run {current_run}/{total_runs}. {fd(avg)} per run, estimate {fd(eta)} to finish."
