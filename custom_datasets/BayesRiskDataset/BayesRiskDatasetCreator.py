from torch.utils.data import Dataset

from misc.PathManager import get_path_manager
import pyarrow as pa
import pyarrow.parquet as pq


class BayesRiskDatasetCreator(Dataset):

    def __init__(self, split, n_hypotheses, n_references, sampling_method, utility, develop=False,
                 base='predictive/tatoeba-de-en/data/raw/', ):
        super().__init__()

        self.split = split

        self.n_hypotheses = n_hypotheses
        self.n_references = n_references

        self.sampling_method = sampling_method

        self.develop = develop

        self.base = base

        self.utility = utility

        self.path_manager = get_path_manager()

        self.columns = [
            "source", 'target', 'hypotheses', 'utilities', 'utilities_count', "count"

        ]

        self.data = {col: [] for col in self.columns}

        self.split = split

    def __len__(self):
        return len(self.data["source"])

    def add_row(self, source, target, hypotheses, utilities, utilities_count, count):
        self.data['source'].append(source)
        self.data['target'].append(target)
        self.data['hypotheses'].append(hypotheses)
        self.data["utilities"].append(utilities)
        self.data["utilities_count"].append(utilities_count)

        self.data['count'].append(count)

    def __getitem__(self, item):
        return {col: self.data[col][item] for col in self.columns}

    def get_dataset_path(self, ):
        relative_path = "{}/{}/{}_{}_scores_{}_{}".format(self.base, self.utility, self.split, self.sampling_method,
                                                          self.n_hypotheses, self.n_references, )
        if self.develop:
            relative_path += '_develop'
        relative_path += '.parquet'

        return self.path_manager.get_abs_path(relative_path)

        # Save the dataset

    def save(self):
        table = pa.Table.from_pydict(self.data)

        dataset_name = self.get_dataset_path()
        pq.write_table(table, dataset_name)
