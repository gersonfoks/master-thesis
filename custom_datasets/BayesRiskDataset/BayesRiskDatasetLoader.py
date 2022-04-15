from datasets import Dataset

from custom_datasets.BayesRiskDataset.BayesRiskDataset import BayesRiskDataset

from misc.PathManager import get_path_manager

import pyarrow.parquet as pq
import numpy as np


class BayesRiskDatasetLoader:
    def __init__(self, split, n_hypotheses, n_references, sampling_method, utility, develop=False,
                 base='predictive/tatoeba-de-en/data/raw/', ):
        self.split = split

        self.n_hypotheses = n_hypotheses
        self.n_references = n_references

        self.sampling_method = sampling_method

        self.develop = develop

        self.base = base

        self.utility = utility

        self.path_manager = get_path_manager()

        self.dataset = None

    def load(self, type="pydict", top_n=0):
        path = self.get_dataset_path()

        if type == "pydict":
            table = pq.read_table(path)
            self.dataset = BayesRiskDataset(table.to_pydict(), self.split, self.n_hypotheses)
        elif type == "pandas":
            table = pq.read_table(path)
            df = table.to_pandas()
            if top_n > 0:
                print("adding top n")
                df["top_n_hypotheses"] = self.add_top_n_to_data(df, top_n)

            self.dataset = BayesRiskDataset(df, self.split, self.n_hypotheses)
        else:
            raise ValueError("no a valid type: {}, choose from pydict, pandas".format(type))
        return self.dataset

    def add_top_n_to_data(self, df, top_n=3):
        def add_top_n_references(x, n=3):
            hypotheses = x["hypotheses"]
            count = x["count"]

            sorted_indices = np.argsort(count)[::-1]
            sorted_count = count[sorted_indices]

            sorted_hypotheses = hypotheses[sorted_indices]

            length = len(sorted_count)
            available_top_n = min(n, length)
            reference_hyp = sorted_hypotheses[:top_n].tolist()
            if available_top_n < top_n:
                reference_hyp += ['<unk>'] * (top_n - available_top_n) # We simple add an unk token if there are not enough hypotheses

            return reference_hyp

        return df.apply(lambda x: add_top_n_references(x, top_n), axis=1)

    def load_as_huggingface_dataset(self):
        path = self.get_dataset_path()
        return Dataset.from_parquet(path)

    def get_dataset_path(self, ):
        relative_path = "{}/{}/{}_{}_scores_{}_{}".format(self.base, self.utility, self.split, self.sampling_method,
                                                          self.n_hypotheses, self.n_references, )
        if self.develop:
            relative_path += '_develop'
        relative_path += '.parquet'

        return self.path_manager.get_abs_path(relative_path)
