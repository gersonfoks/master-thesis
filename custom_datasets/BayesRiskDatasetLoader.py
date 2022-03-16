import pandas as pd

from custom_datasets.BayesRiskDataset import BayesRiskDataset

from utils.PathManager import get_path_manager

import pyarrow.parquet as pq

import pyarrow as pa

'''
Code for loading bayes risk datasets.

The format of the datasets are:
{split}_predictive_{sample_method}_scores_{n_hypothesis}_{n_references}
'''


class BayesRiskDatasetLoader:
    def __init__(self, split, n_hypotheses, n_references, sampling_method, develop=False, base='predictive/tatoeba-de-en/data/raw/', ):
        self.split = split

        self.n_hypotheses = n_hypotheses
        self.n_references = n_references

        self.sampling_method = sampling_method

        self.develop = develop

        self.base = base

        self.path_manager = get_path_manager()

        self.dataset = None

    def load(self):
        path = self.get_dataset_path()
        table = pq.read_table(path)

        self.dataset = BayesRiskDataset(table.to_pydict(), self.split, self.n_hypotheses)
        return self.dataset

    def load_empty(self):
        '''
        Loads an empty dataset
        :return:
        '''
        self.dataset = BayesRiskDataset(None, self.split)
        return self.dataset

    def get_dataset_path(self, ):
        relative_path = "{}{}_{}_scores_{}_{}".format(self.base, self.split, self.sampling_method,
                                                      self.n_hypotheses, self.n_references, )
        if self.develop:
            relative_path += '_develop'
        relative_path += '.parquet'


        return self.path_manager.get_abs_path(relative_path)

    # Save the dataset
    def save(self):
        table = pa.Table.from_pydict(self.dataset.data)

        dataset_name = self.get_dataset_path()
        pq.write_table(table, dataset_name)


