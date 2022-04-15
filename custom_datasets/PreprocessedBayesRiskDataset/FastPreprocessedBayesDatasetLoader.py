import os


from custom_datasets.CachedTable import CachedTable
from custom_datasets.PreprocessedBayesRiskDataset.FastPreBayesDataset import FastPreBayesDataset
from custom_datasets.utils import load_json_as_dict, load_csv_as_df
from misc.PathManager import get_path_manager


class FastPreBayesDatasetLoader:

    def __init__(self, dataset_location, split, features, max_tables=50, develop=False,
                 repeated_indices=True, on_hpc=False):
        self.dataset_location = dataset_location
        self.split = split  # Amount of rows
        self.features = features
        self.max_tables = max_tables
        self.develop = develop
        self.repeated_indices = repeated_indices

        self.dataset = None

        self.path_manager = get_path_manager()

        self.on_hpc = on_hpc
        self.hpc_path_manager = get_path_manager('scratch')

    def load(self):

        # If it is already loaded we return the one we already loaded.
        if self.dataset:
            return self.dataset

        indices = self.get_indices()

        cached_tables = self.get_cached_dataset()

        return FastPreBayesDataset(indices, cached_tables, max_loaded_datasets=self.max_tables)

    def get_main_dir(self):
        ref = self.dataset_location + self.split + '/'
        if self.develop:
            ref += 'develop/'

        return self.path_manager.get_abs_path(ref)

    def get_table_dir(self):
        if not self.on_hpc:
            return self.get_main_dir()
        else:
            ref = self.dataset_location + self.split + '/'
            return self.hpc_path_manager.get_abs_path(ref)

    def get_metadata(self):
        path = self.get_main_dir()
        path += 'metadata.json'

        return load_json_as_dict(path)

    def get_indices(self):
        path = self.get_main_dir()
        path += 'indices_count.csv'
        metadata = self.get_metadata()
        indices_count = load_csv_as_df(path)
        batched_indices = [indices_count.loc[start:end - 1] for start, end in metadata["dataset_indices"]]
        if self.repeated_indices:
            new_batched_indices = []
            for df in batched_indices:
                df = df.reindex(df.index.repeat(df["count"]))
                df = df.reset_index(level=0, inplace=False, drop=True)
                new_batched_indices.append(df['id'].to_list())
            batched_indices = new_batched_indices
        else:
            batched_indices = [b["id"].to_list() for b in batched_indices]
        return batched_indices

    def get_cached_dataset(self):
        path = self.get_table_dir() + 'data/'
        files = os.listdir(path)
        files = sorted(files, key=lambda x: int(x[:-6]))
        cached_tables = []
        metadata = self.get_metadata()
        for t in range(len(files)):
            table_ref = path + '/' + files[t]
            start_id = metadata["dataset_indices"][t][0]
            end_id = metadata["dataset_indices"][t][1]
            cached_table = CachedTable(table_ref, start_id, end_id,
                                       self.features)
            cached_tables.append(cached_table)
        return cached_tables
