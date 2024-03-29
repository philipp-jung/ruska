import os
import json
import logging
import datetime
import itertools
from multiprocessing import Pool
from pathlib import Path, PosixPath
from pprint import pprint
from pathlib import Path
from typing import Dict, List, Callable, Union

from ruska.helpers import send_notification, estimate_time_to_finish


class Ruska:
    """
    When running data-cleaning experiments, one needs to be extremely diligent to not
    mess up the measurements. This reminds me of my time studying Physics. Measuring
    experiments, I needed to be very careful and diligent, too.

    This is why there is Ruska: Ernst Ruska was a experimental phsicist, and the first
    to construct an electron microscope. He needed to be ridiculously careful building
    that thing in 1933.

    I need a similar level of diligence keeping track of my measurements. Ruska keeps
    track of parameters and measurements.
    """

    def __init__(
        self,
        name: str,
        description: str,
        commit: str,
        config: dict,
        ranges: Dict[str, list],
        runs: int,
        save_path: str,
        chat_id: Union[None, str] = None,
        token: Union[None, str] = None,
        is_logging: bool = True,
    ):
        """Pass all parameters for raha as kwargs."""
        self.name = name
        self.description = description
        self.commit = commit
        self.config = config
        self.ranges = {**ranges, "run": list(range(runs))}
        self.save_path = Path(save_path) / f"{name}.txt"
        self.chat_id = chat_id
        self.token = token
        self.times = []

        for range_key in ranges:
            if range_key not in config.keys():
                raise ValueError("Ranges müssen im Config dict enthalten sein.")

        self.range_combinations: List[dict] = []

        if is_logging:
            self.logging_path = os.path.splitext(self.save_path)[0] + ".log"
            logger = logging.getLogger("ruska")
            logger.info(f"Writing logs to {self.logging_path}.")

    @property
    def start_time(self):
        return self.times[0]

    @property
    def end_time(self):
        return self.times[-1]

    def _combine_ranges(self) -> None:
        """
        Calculate all possible combinations of ranges. Assigns them to
        self.range_combinations.
        @return: None
        """
        value_combinations = itertools.product(*list(self.ranges.values()))
        for combination in value_combinations:
            range_combinations = {}
            for i, key_range in enumerate(self.ranges.keys()):
                range_combinations[key_range] = combination[i]
            self.range_combinations.append(range_combinations)

    def run(self, experiment: Callable, parallel=False, workers=None):
        self._combine_ranges()

        # overwrite config with range when specified
        configs = [
            {**self.config, **range_config} for range_config in self.range_combinations
        ]
        logger = logging.getLogger(__name__)
        logger.debug(f'Generated configs \n {json.dumps(configs, indent=2)}')

        self.times.append(datetime.datetime.now())

        send_notification(
            f"Ruska starts an experiment called {self.name}.", self.chat_id, self.token
        )

        results = []
        if parallel:
            pool = Pool(workers)
            results = pool.starmap(experiment, enumerate(configs))
            pool.close()
            self.times.append(datetime.datetime.now())
        else:
            for i, config in enumerate(configs):
                try:
                    result = experiment(i, config)
                except Exception as e:
                    result = e
                results.append(result)
                self.times.append(datetime.datetime.now())
                print(estimate_time_to_finish(self.times, len(self.range_combinations)))

        logger.info(f'Finished {len(configs)} measurements.')

        config_store = {
            k: v
            for k, v in vars(self).items()
            if k not in ["range_combinations", "token", "chat_id"]
        }

        with open(self.save_path, "w") as f:
            print("Experiment using Ruska finished.", file=f)
            print(f"Start time: {self.times[0]} -- End time: {self.times[-1]}", file=f)
            print("Ruska was configured as follows:", file=f)
            print("[BEGIN CONFIG]", file=f)
            pprint(config_store, f)
            print("[END CONFIG]", file=f)
            print("Ruska measured the following results:", file=f)
            print("[BEGIN RESULTS]", file=f)
            pprint(results, f)
            print("[END RESULTS]", file=f)
        print("Measurement finished")
        send_notification(
            f"Measurements of experiment {self.name} finished.\n"
            f"Results are stored at {self.save_path}.",
            self.chat_id,
            self.token,
        )
        logger.info(f'Wrote results to {self.save_path}. Stopping.')

    @staticmethod
    def load_result(path_to_result: str):
        """
        Loads a ruska result and returns a tuple result_dict, config_dict.
        """
        path = Path(path_to_result)
        config_flag = False
        result_flag = False

        result = ""
        result_config = ""

        with open(path, "rt") as f:
            for line in f:
                if line.strip() == "[BEGIN CONFIG]":
                    config_flag = True
                elif line.strip() == "[END CONFIG]":
                    config_flag = False
                elif line.strip() == "[BEGIN RESULTS]":
                    result_flag = True
                elif line.strip() == "[END RESULTS]":
                    result_flag = False
                else:
                    if config_flag:
                        result_config = result_config + line
                    elif result_flag:
                        result = result + line
        result_dict = eval(result)  # this is where PosixPath is used
        result_config_dict = eval(result_config)
        return result_dict, result_config_dict
