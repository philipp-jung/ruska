import random
import pandas as pd
from pathlib import Path
from jenga.corruptions.generic import CategoricalShift
from ruska.helpers import simple_mcar


class Corruptor:
    """
    Run Jenga to corrupt a dataset.
    """

    def __init__(self, dataset_name: str):
        self.dataset_path = Path(dataset_name + ".csv")
        self.export_root = Path(dataset_name)
        self.samplings = ["MCAR", "MAR", "MNAR"]
        self.fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    def run(self):
        df = pd.read_csv(self.dataset_path, sep=",")
        df = df.astype(str)

        for s in self.samplings:
            for f in self.fractions:
                df_dirty = df.copy()
                for column in df.columns:
                    df_dirty[column] = CategoricalShift(
                        column=column, fraction=f, sampling=s
                    ).transform(df_dirty)[column]
                export_path = self.export_root / Path(s)
                export_path.mkdir(parents=True, exist_ok=True)
                formatted_fraction = str(f).split(".")[1]
                df_dirty.to_csv(
                    export_path / f"dirty_{formatted_fraction}.csv", index=True
                )
        df.to_csv(self.export_root / "clean.csv", index=True)

    def run_simple_mcar(self):
        """
        Jenga is buggy, so this is my stupid MCAR implementation. I used
        this in 2022W22. It casts the entire DataFrame to string, and is thus
        not applicable when working with the imputer feature generator.
        """
        df = pd.read_csv(self.dataset_path, sep=",")
        df = df.astype(str)

        for f in self.fractions:
            df_dirty = simple_mcar(df, f, error_token='ERRORABC123!?')
            export_path = self.export_root / Path("MCAR/")
            export_path.mkdir(parents=True, exist_ok=True)
            formatted_fraction = str(f).split(".")[1]
            df_dirty.to_csv(export_path / f"dirty_{formatted_fraction}.csv", index=True)

        df.to_csv(self.export_root / "clean.csv", index=True)
