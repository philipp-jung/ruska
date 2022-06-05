import random
import pandas as pd
from pathlib import Path
from jenga.corruptions.generic import CategoricalShift


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
        """Jenga is buggy, so this is my stupidly MCAR implementation."""
        df = pd.read_csv(self.dataset_path, sep=",")
        df = df.astype(str)

        for f in self.fractions:
            df_dirty = df.copy()
            n_rows, n_cols = df.shape

            target_corruptions = round(n_rows * n_cols * f)
            error_cells = random.sample(
                [(x, y) for x in range(n_rows) for y in range(n_cols)],
                k=target_corruptions,
            )

            for x, y in error_cells:
                df_dirty.iat[x, y] = "ERRORABC123!?"

            export_path = self.export_root / Path("MCAR/")
            export_path.mkdir(parents=True, exist_ok=True)
            formatted_fraction = str(f).split(".")[1]
            df_dirty.to_csv(export_path / f"dirty_{formatted_fraction}.csv", index=True)

        df.to_csv(self.export_root / "clean.csv", index=True)
