import ast
from datetime import datetime

from torch.utils.data import Dataset
import torch
import numpy as np


class CachedBayesRiskDataset(Dataset):
    '''
    Class loads chunks of the data before. It has a custom shuffle function
    This turns to have the lowest overhead when handling data. Making it relatively fast to use.
    '''
    def __init__(self, data, ref_table, split, feature_names, chunk_size=4096 * 32):
        super().__init__()

        self.data = data
        self.ref_table = ref_table
        self.split = split
        self.feature_names = feature_names

        # THis maps an index to an id. THis is used to "lazily" shuffle the dataset. We shuffle the dataset this way to
        self.ids_map = list(range(0, len(self.data)))

        self.ids_to_index = {}

        self.chunk_size = chunk_size

        self.current_start_id = -1
        self.current_end_id = -1

        self.first_load = True

        self.shuffle()

    def __len__(self):
        return len(self.data)

    def shuffle(self):
        # Shuffles the dataset (lazely shuffles the dataset)

        np.random.shuffle(self.ids_map)

    def load_chunk(self, start_id, end_id):

        ids = self.ids_map[start_id: end_id]

        self.current_start_id = start_id
        self.current_end_id = end_id

        data = self.data[ids]

        ref_ids = data["ref_id"]

        # For taking the ref ids
        self.ids_to_index = {
            i: index for index, i in enumerate(ids)
        }

        # Get it out of the table

        self.preloaded_data = self.ref_table.take(ref_ids)

        # Make it ready

        source_hyp = {k: self.preloaded_data[k] for k in ["sources", "hypothesis", ]}
        util_count = {k: self.preloaded_data[k].to_numpy() for k in ["utilities", "utilities_count"]}

        temp_features = {
            feature_name: torch.Tensor(np.stack(self.preloaded_data[feature_name].to_numpy())) for feature_name in
            self.feature_names
        }

        self.preloaded_data = {**temp_features, **source_hyp, **util_count}

    def load_next_chunk(self):
        print("loading next chunk")

        if self.first_load:
            next_start_id = 0
            next_end_id = min(self.chunk_size, len(self))
            self.load_chunk(next_start_id, next_end_id)
            self.first_load = False

        else:
            next_start_id = self.current_start_id + self.chunk_size

            next_end_id = next_start_id + self.chunk_size

            if next_start_id > len(self):
                next_start_id = 0
                next_end_id = self.chunk_size

            # Make sure we never go over the total length of the data
            next_end_id = min(next_end_id, len(self))

            self.load_chunk(next_start_id, next_end_id)

    def __getitem__(self, idx):

        if idx >= self.current_end_id:
            # Load next chunk
            self.load_next_chunk()
        # This means that we need to start over.
        if idx < self.current_start_id:
            self.load_next_chunk()

        shuffled_id = self.ids_map[idx]

        # Get the ref ids

        list_index = self.ids_to_index[shuffled_id]

        source = self.preloaded_data["sources"][list_index]
        hypothesis = self.preloaded_data["hypothesis"][list_index]
        features = {feature_name: self.preloaded_data[feature_name][list_index] for feature_name in self.feature_names}

        utilities = self.preloaded_data["utilities"][list_index]
        utilities_count = self.preloaded_data["utilities_count"][list_index]

        item = {**features, "source": source, "hypothesis": hypothesis, 'utilities': utilities,
                'utilities_count': utilities_count}

        return item
