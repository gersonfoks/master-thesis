import os


from torch.utils.data import Dataset

import numpy as np


from custom_datasets.CachedTable import CachedTable
from custom_datasets.utils import load_json_as_dict, load_csv_as_df
from utils.PathManager import get_path_manager


class FastPreBayesDataset(Dataset):

    def __init__(self, table_indices, cached_datasets, max_loaded_datasets=3):
        self.table_indices = table_indices

        self.tables = list(range(len(cached_datasets)))

        self.cached_datasets = cached_datasets

        self.max_loaded_datasets = max_loaded_datasets
        # apparently we can load the whole dataset into ram without issues so we will do that:
        self.mapping = None
        self.create_mapping()

        self.current_datasets = []
        self.first_load = True
        self.current_dataset_start_id = 0
        self.current_dataset_end_id = 0

        self.current_tables = []

    def load_dataset(self, dataset_id):
        self.cached_datasets[dataset_id].load()

    def unload_dataset(self, dataset_id):
        self.cached_datasets[dataset_id].unload()

    def __len__(self):
        return np.sum([len(i) for i in self.table_indices])

    def create_mapping(self):
        self.mapping = []
        for t in self.tables:
            indices = self.table_indices[t]
            self.mapping += [(t, idx) for idx in indices]

    def shuffle(self):
        print("shuffle")
        np.random.shuffle(self.tables)
        # Shuffle the indices
        for indices in self.table_indices:
            np.random.shuffle(indices)
        # Next we create a mapping
        self.create_mapping()


    def load_next_datasets(self):

        if self.first_load == True:

            self.first_load = False
            self.current_dataset_start_id = 0
            self.current_dataset_end_id = self.max_loaded_datasets


        else:
            for dataset in self.current_datasets:
                dataset.unload()

            self.current_dataset_start_id = self.current_dataset_end_id
            self.current_dataset_end_id = self.current_dataset_end_id + self.max_loaded_datasets

        self.current_datasets = []


        # Check if we need to start over
        if self.current_dataset_start_id == len(self.tables):
            self.current_dataset_start_id = 0
            self.current_dataset_end_id = self.max_loaded_datasets

        self.current_dataset_end_id = min(self.current_dataset_end_id, len(self.tables))

        self.load_datasets(self.current_dataset_start_id, self.current_dataset_end_id)

    def load_datasets(self, start_id, end_id):
        self.current_tables = []
        for i in range(start_id, end_id):
            t = self.tables[i]
            print("loading:", t)
            self.load_dataset(t)
            self.current_datasets.append(self.cached_datasets[t])

            self.current_tables.append(t)

    def __getitem__(self, item):

        # get the indices id

        t, idx = self.mapping[item]

        # Check if we need to start loading the next tables
        if t not in self.current_tables:
            self.load_next_datasets()
        try:
            r = self.cached_datasets[t][idx]

        except IndexError as e:
            print(item)
            print(t)
            print(idx)
            print(self.mapping)
            print(self.current_tables)
            raise e
        return r


class FastPreBayesDatasetLoader:

    def __init__(self, dataset_location, split, features, max_tables=50, develop=False,
                 repeated_indices=True):
        self.dataset_location = dataset_location
        self.split = split  # Amount of rows
        self.features = features
        self.max_tables = max_tables
        self.develop = develop
        self.repeated_indices = repeated_indices

        self.dataset = None

        self.path_manager = get_path_manager()

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
        path = self.get_main_dir() + '/data/'
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
