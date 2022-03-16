from custom_datasets.utils import load_arrow_file, load_arrow_file_in_memory
import numpy as np

class CachedTable:

    def __init__(self, ref, start_id, end_id, features):
        self.ref = ref
        self.start_id = start_id
        self.end_id = end_id
        self.features = features
        self.preloaded_data = {**{feature_name: [] for feature_name in self.features}, "source": [], "hypothesis": [],
                               "utilities": [], "utilities_count": []}



    def __len__(self):
        return self.end_id - self.start_id

    def load(self):
        print("load cached table")
        data = load_arrow_file_in_memory(self.ref)
        #data = data_temp.take(list(range(len(data_temp))))
        self.preloaded_data["source"] += list(data["source"].to_pylist())
        self.preloaded_data["hypothesis"] += list(data["hypotheses"].to_pylist())
        self.preloaded_data["utilities"] = np.array(data["utilities"].to_numpy())



        for feature_name in self.features:
            d = data[feature_name].to_numpy()

            self.preloaded_data[feature_name] = d

    def unload(self):
        del self.preloaded_data

        self.preloaded_data = {**{feature_name: [] for feature_name in self.features}, "source": [], "hypothesis": [],
                               "utilities": [], "utilities_count": []}

    def __getitem__(self, idx):
        # print("Getting item: start, idx, end")
        # print(self.start_id)
        # print(idx)
        # print(self.end_id)
        relative_id = idx - self.start_id

        if idx < self.start_id or idx > self.end_id:
            raise ValueError("idx not in range: ", (self.start_id, self.end_id, idx))


        source = self.preloaded_data["source"][relative_id]
        hypothesis = self.preloaded_data["hypothesis"][relative_id]
        features = {feature_name: self.preloaded_data[feature_name][relative_id] for feature_name in self.features}

        utilities = self.preloaded_data["utilities"][relative_id]
        #utilities_count = self.preloaded_data["utilities_count"][relative_id]

        item = {**features, "source": source, "hypothesis": hypothesis, 'utilities': utilities,
                }

        return item