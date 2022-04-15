
import ast


from custom_datasets.SampleDataset.SampleDataset import SampleDataset
from misc.PathManager import get_path_manager
import pandas as pd

class SampleDatasetLoader:
    def __init__(self, split, n_samples, sampling_method, develop=False, base='/NMT/tatoeba-de-en'):
        self.split = split

        self.n_samples = n_samples

        self.sampling_method = sampling_method

        self.develop = develop

        self.base = base

        self.path_manager = get_path_manager()

        self.dataset = None

    def load(self):

        # If it is already loaded we return the one we already loaded.
        if self.dataset:
            return self.dataset
        path = self.get_dataset_path()

        df = pd.read_csv(path, sep="\t")

        df["samples"] = df["samples"].map(lambda x: ast.literal_eval(x))
        df["count"] = df["count"].map(lambda x: ast.literal_eval(x))

        self.dataset = SampleDataset(data=df.to_dict(), split=self.split)
        return self.dataset

    def get_dataset_path(self):
        if self.develop:

            save_file = "{}{}_{}_{}_develop.csv".format(self.base, self.split, self.sampling_method,
                                                        self.n_samples, )
        else:
            save_file = "{}{}_{}_{}.csv".format(self.base, self.split, self.sampling_method,
                                                        self.n_samples, )

        save_file = self.path_manager.get_abs_path(save_file)
        return save_file
