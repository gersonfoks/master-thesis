import os

import psutil
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from custom_datasets.utils import load_json_as_dict, load_csv_as_df, load_arrow_file
from utils.PathManager import get_path_manager


class PreBayesRiskDataset(Dataset):

    def __init__(self, indices, start_end_indices, features, dataset_location, max_chunk_size=4096 * 8,
                 split="train_predictive", develop=False):
        # Create a mapping from indices to sub datasets
        self.develop = develop
        self.tables = None

        self.split = split

        self.features = features

        self.max_chunk_size = max_chunk_size

        self.indices = indices
        self.start_end_indices = start_end_indices

        self.length = np.sum([len(t) for t in indices])

        self.current_start_id = 0
        self.current_end_id = 0

        self.active_sub_dataset = None

        self.preloaded_data = None

        self.dataset_location = dataset_location

        self.first_load = True
        self.path_manager = get_path_manager()
        self.construct_table_id_map()

        self.shuffle()

    def construct_table_id_map(self):
        self.table_id_map = []
        for i, indices in enumerate(self.indices):
            self.table_id_map += [(i, x) for x in indices]

    def __len__(self):
        return self.length

    def shuffle(self):

        np.random.shuffle(self.table_id_map)


    def load_next_chunk(self):
        print("loading next chunk")
        del self.preloaded_data

        self.reload_tables()

        if self.first_load:
            next_start_id = 0
            next_end_id = min(self.max_chunk_size, len(self))

            self.preloaded_data = self.load_chunk(next_start_id, next_end_id)
            self.first_load = False

        else:
            next_start_id = self.current_start_id + self.max_chunk_size

            next_end_id = next_start_id + self.max_chunk_size

            if next_start_id > len(self):
                next_start_id = 0
                next_end_id = self.max_chunk_size

            # Make sure we never go over the total length of the data
            next_end_id = min(next_end_id, len(self))

            self.preloaded_data = self.load_chunk(next_start_id, next_end_id)

    def load_chunk(self, start_id, end_id):

        # First delete the old chunk

        ids = self.table_id_map[start_id: end_id]

        tables_to_take = {i: [] for i in range(len(self.tables))}
        original_map = {i: [] for i in range(len(self.tables))}
        for i, (table_i, id) in enumerate(ids):

            tables_to_take[table_i].append(id)
            original_map[table_i].append(i)

        # We need to map in the proper way back
        self.resulting_map = []
        for ids in original_map.values():
            self.resulting_map += ids



        preloaded_data = {**{feature_name: [] for feature_name in self.features}, "source": [], "hypothesis": [],
                          "utilities": [], "utilities_count": []}

        for i, table_to_take in tables_to_take.items():

            if len(table_to_take) > 0:

                data = self.tables[i].take(table_to_take)

                preloaded_data["source"] += data["source"].to_pylist()
                preloaded_data["hypothesis"] += data["hypotheses"].to_pylist()
                preloaded_data["utilities"].append(data["utilities"].to_numpy())

                # preloaded_data["utilities_count"].append(data["utilities_count"].to_numpy())

                for feature_name in self.features:
                    d = data[feature_name].to_numpy()
                    preloaded_data[feature_name].append(d)

                x = self.tables[i]
                self.tables[i] = []
                del x

        preloaded_data["utilities"] = np.concatenate(preloaded_data["utilities"], axis=0)
        # preloaded_data["utilities_count"] = np.concatenate(preloaded_data["utilities_count"], axis=0)
        for feature_name in self.features:
            preloaded_data[feature_name] = np.concatenate(preloaded_data[feature_name], axis=0)

        self.current_end_id = end_id
        self.current_start_id = start_id

        return preloaded_data

    def __getitem__(self, idx):
        # Getting an item is done as follows:
        # First we map it to

        if idx >= self.current_end_id or idx < self.current_start_id:
            self.load_next_chunk()

        # Get the relative id
        relative_id = idx - self.current_start_id


        # We also need to map it to the right id
        relative_id = self.resulting_map[relative_id]



        source = self.preloaded_data["source"][relative_id]
        hypothesis = self.preloaded_data["hypothesis"][relative_id]
        features = {feature_name: self.preloaded_data[feature_name][relative_id] for feature_name in self.features}

        utilities = self.preloaded_data["utilities"][relative_id]
        # utilities_count = self.preloaded_data["utilities_count"][relative_id]

        item = {**features, "source": source, "hypothesis": hypothesis, 'utilities': utilities,
                } #'utilities_count': utilities_count

        return item

    def reload_tables(self):
        self.tables = self.get_tables()

    def get_main_dir(self):
        ref = self.dataset_location + self.split + '/'
        if self.develop:
            ref += 'develop/'



        return self.path_manager.get_abs_path(ref)

    def get_tables(self):
        path = self.get_main_dir()
        path += 'data/'

        # Loop over the datasets files
        files = os.listdir(path)

        # Sort them

        files = sorted(files, key=lambda x: int(x[:-6]))

        # Load the arrow files
        tables = []
        for file in files:
            file_path = path + file
            table = load_arrow_file(file_path)

            tables.append(table)

        return tables


class PreBayesRiskDatasetLoader:
    def __init__(self, dataset_location, split, features, max_chunk_size=4096 * 8, develop=False, repeated_indices=True):
        self.dataset_location = dataset_location
        self.split = split  # Amount of rows
        self.features = features
        self.max_chunk_size = max_chunk_size
        self.develop = develop
        self.repeated_indices = repeated_indices

        self.dataset = None



        self.path_manager = get_path_manager()

    def load(self):

        # If it is already loaded we return the one we already loaded.
        if self.dataset:
            return self.dataset

        metadata = self.get_metadata()

        start_end_indices = metadata["dataset_indices"]

        indices = self.get_indices()

        # Next we split the indices and expand if needed

        indices_expanded = []

        for start, end in start_end_indices:
            temp = {}
            temp["ids"] = np.array(indices["id"][start: end].to_list()) - start
            temp["count"] = indices["count"][start: end].to_list()
            # Create the dataframe
            df = pd.DataFrame.from_dict(temp)

            # If we want to allow for repeated indices ()
            if self.repeated_indices:
                df = df.reindex(df.index.repeat(df["count"]))
                df = df.reset_index(level=0, inplace=False, drop=True)


            indices_expanded.append(df['ids'].to_list())

        # Create the dataset
        return PreBayesRiskDataset(indices_expanded, start_end_indices, self.features,
                                   dataset_location=self.dataset_location,
                                   max_chunk_size=self.max_chunk_size, split=self.split, develop=self.develop)

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
        return load_csv_as_df(path)
