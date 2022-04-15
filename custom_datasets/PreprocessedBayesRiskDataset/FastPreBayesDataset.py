from torch.utils.data import Dataset

import numpy as np


class FastPreBayesDataset(Dataset):

    def __init__(self, table_indices, cached_datasets, max_loaded_datasets=3):


        # A list containing which indices are there in each
        self.table_indices = table_indices


        # Indices of the tables
        self.tables = list(range(len(cached_datasets)))

        # The actual cached datasets
        self.cached_datasets = cached_datasets

        # The maximum amount of datasets we may have in the memory at any given time.
        self.max_loaded_datasets = max_loaded_datasets

        # We need to create a mapping from the indices to the tables we have.
        self.create_mapping()

        self.current_datasets = []

        # It is the first time we load the dataset.
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
        # This mapping keeps track of where we should get the indices from.
        # The mapping contains tuples, the table index and the (absolute) index of the
        self.mapping = []
        for t in self.tables:
            indices = self.table_indices[t]
            self.mapping += [(t, idx) for idx in indices]

    def shuffle(self):
        # Shuffle the tables around
        np.random.shuffle(self.tables)
        # Shuffle the indices for each table around
        for indices in self.table_indices:
            np.random.shuffle(indices)
        # Next we create a mapping
        self.create_mapping()

    def load_next_datasets(self):

        if self.first_load == True:
            # The first time we load we need to set dataset ids as follows:
            self.first_load = False
            self.current_dataset_start_id = 0
            self.current_dataset_end_id = self.max_loaded_datasets


        else:
            # We first unload all the datasets
            for dataset in self.current_datasets:
                dataset.unload()

            # Next we set the current dataset ids
            self.current_dataset_start_id = self.current_dataset_end_id
            self.current_dataset_end_id = self.current_dataset_end_id + self.max_loaded_datasets

            # Check if we need to start over
            if self.current_dataset_start_id >= len(self.tables):
                self.current_dataset_start_id = 0
                self.current_dataset_end_id = self.max_loaded_datasets

        self.current_datasets = []
        self.current_dataset_end_id = min(self.current_dataset_end_id, len(self.tables))

        self.load_datasets(self.current_dataset_start_id, self.current_dataset_end_id)

    def load_datasets(self, start_id, end_id):

        # Load the given datasets
        self.current_tables = []
        for i in range(start_id, end_id):
            t = self.tables[i]
            self.load_dataset(t)
            self.current_datasets.append(self.cached_datasets[t])

            self.current_tables.append(t)

    def __getitem__(self, item):
        # We assume that we iterate through the data one by one, such that we can plan ahead and load/unload data as needed.
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
            print(self.current_tables)
            raise e
        return r
