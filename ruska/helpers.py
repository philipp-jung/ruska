import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List


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


def jenga_plot(*, ruska_result_pdep, ruska_result_naive):
    """
    A bar plot, comparing cleaning f1-scores between pdep and naive vicinity
    model, grouped by error rate.
    """
    result_pdep = [{**x["config"], **x["result"]} for x in ruska_result_pdep]
    result_naive = [{**x["config"], **x["result"]} for x in ruska_result_naive]

    r_pdep = reduce_runs(result_pdep, run_label="run")
    r_naive = reduce_runs(result_naive, run_label="run")

    samplings = get_distinct_list(x["sampling"] for x in result_pdep)
    error_fractions = get_distinct_list(x["error_fraction"] for x in result_pdep)
    fig, axs = plt.subplots(len(samplings), 1, figsize=(14, 8))
    axs = np.ravel(axs)
    rects = []
    for i, sampling in enumerate(samplings):

        pdep = [x for x in r_pdep if (x["sampling"] == sampling)]
        naive = [x for x in r_naive if (x["sampling"] == sampling)]

        x = np.arange(len(error_fractions))
        width = 0.3

        rects0 = axs[i].bar(
            x - width / 2,
            [round(x["f1_avg"], 2) for x in pdep],
            width,
            label="Pdep",
            color="C0",
            yerr=[x["f1_se"] for x in pdep],
        )
        rects1 = axs[i].bar(
            x + width / 2,
            [round(x["f1_avg"], 2) for x in naive],
            width,
            label="Naive",
            color="C1",
            yerr=[x["f1_se"] for x in naive],
        )

        axs[i].set_title(f"Sampling Strategy {sampling}")
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(error_fractions)
        axs[i].legend()

        axs[i].set_xlabel("Error Rate")
        axs[i].set_ylabel("Cleaning F1-Score")

        axs[i].bar_label(rects0, padding=3)
        axs[i].bar_label(rects1, padding=3)
        axs[i].legend(loc="upper left")

    fig.tight_layout()
    fig.suptitle(
        "Breast-Cancer Dataset, three Corruption Strategies, Simple Value Deletion",
        fontsize=16,
    )
    fig.subplots_adjust(top=0.9)
    # plt.savefig('./4-cat-shift-adult-all-sampling-strategies-all-rates.png')
    return fig
