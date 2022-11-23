import math
from .helpers import reduce_runs, get_distinct_list
from matplotlib import pyplot as plt
import numpy as np


def prepare_result_v1(ruska_result: tuple):
    """
    Transform what is returned by Ruska.load_result() into a format that is
    more handy to work with. I use it for plotting, but it's more handy in other
    cases, too.
    First used in 2022W46.
    """
    result, ruska_config = ruska_result
    exp_config = ruska_config['config']
    full_result = [{**x['config'], **x['result']} for x in result]
    superfluous_config = [x for x in list(exp_config.keys()) if x not in ruska_config['ranges']]
    formatted_result = [{k: v for k, v in x.items() if k not in superfluous_config} for x in full_result]
    return formatted_result, ruska_config


def plot_bars(formatted_result,
                      ruska_config,
                      parameter: str,
                      run_label: str = 'run',
                      score: str = 'f1',
                      title=None):
    """
    Take prepared results from Ruska and plot them in a subplot. The subplot
    contains as many plots as the result contains datasets. One parameter
    can then be plotted on the x-axis,  while the y-axis displays one of the
    three classification scores.
    """
    r = reduce_runs(formatted_result, run_label=run_label)
    datasets = ruska_config['ranges']['dataset']
    n_rows = math.ceil(len(datasets)/2)

    fig, axs = plt.subplots(n_rows, 2, figsize=(17,17))
    axs = np.ravel(axs)
    rects = []
    for i in range(len(axs)):
        y = [round(x[f'{score}_avg'], 2) for x in r if x['dataset'] == datasets[i]]
        x = [str(x[parameter]) for x in r if x['dataset'] == datasets[i]]
        rect = axs[i].bar(x, y, color=f'C{i}')
        rects.append(rect)
        axs[i].set_title(datasets[i])

    for i, ax in enumerate(axs.flat):
        ax.set(xlabel=parameter, ylabel=f'{score}-Score Cleaning')
        ax.bar_label(rects[i], padding=3)

    if title is not None:
        fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    return fig


def jenga_plot(*, ruska_result_pdep, ruska_result_naive, ruska_config):
    """
    A bar plot, comparing cleaning f1-scores between pdep and naive vicinity
    model, grouped by error rate.
    Creates an axis per error-sampling method in the ruska_config.
    """
    result_pdep = [{**x["config"], **x["result"]} for x in ruska_result_pdep]
    result_naive = [{**x["config"], **x["result"]} for x in ruska_result_naive]

    r_pdep = reduce_runs(result_pdep, run_label="run")
    r_naive = reduce_runs(result_naive, run_label="run")

    samplings = get_distinct_list(x["sampling"] for x in result_pdep)
    error_fractions = get_distinct_list(x["error_fraction"] for x in result_pdep)
    fig, axs = plt.subplots(len(samplings), 1, figsize=(14, 8))
    axs = np.ravel(axs)
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
    dataset = ruska_config["config"]["dataset"]
    n_strat = len(ruska_config["ranges"].get("sampling", [1]))
    fig.suptitle(
        f"{dataset} Dataset, {n_strat} Corruption Strategies",
        fontsize=16,
    )
    fig.subplots_adjust(top=0.9)
    return fig


def jenga_plot_datasets(*, ruska_result_pdep, ruska_result_naive, ruska_config):
    """
    A bar plot, comparing cleaning f1-scores between pdep and naive vicinity
    model, grouped by error rate.
    Creates an axis per dataset in the ruska_config. Assumes a static
    error-sampling-method.
    """
    result_pdep = [{**x["config"], **x["result"]} for x in ruska_result_pdep]
    result_naive = [{**x["config"], **x["result"]} for x in ruska_result_naive]

    r_pdep = reduce_runs(result_pdep, run_label="run")
    r_naive = reduce_runs(result_naive, run_label="run")

    datasets = get_distinct_list(x["dataset"] for x in result_pdep)
    error_fractions = get_distinct_list(x["error_fraction"] for x in result_pdep)
    fig, axs = plt.subplots(len(datasets), 1, figsize=(14, 8))
    axs = np.ravel(axs)
    for i, dataset in enumerate(datasets):

        pdep = [x for x in r_pdep if (x["dataset"] == dataset)]
        naive = [x for x in r_naive if (x["dataset"] == dataset)]

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

        axs[i].set_title(f"On dataset {dataset}")
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(error_fractions)
        axs[i].legend()

        axs[i].set_xlabel("Error Rate")
        axs[i].set_ylabel("Cleaning F1-Score")

        axs[i].bar_label(rects0, padding=3)
        axs[i].bar_label(rects1, padding=3)
        axs[i].legend(loc="upper left")

    fig.tight_layout()
    n_datasets = len(datasets)
    n_runs = len(ruska_config["ranges"]["run"])
    fig.suptitle(
        f"{n_datasets} Datasets, {n_runs} Runs per Dataset.",
        fontsize=16,
    )
    fig.subplots_adjust(top=0.9)
    return fig

