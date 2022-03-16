from pathlib import Path

from datasets import Dataset

from custom_datasets.utils import save_arrow_file, save_dict_as_json, save_dict_as_csv

from utils.PathManager import get_path_manager

from pathlib import Path

from datasets import Dataset

from custom_datasets.utils import save_arrow_file, save_dict_as_json, save_dict_as_csv

from utils.PathManager import get_path_manager


class PreBayesDatasetCreator:

    def __init__(self, dataset_location, features, n_hypotheses=10, n_references=100, max_dataset_size=4096 * 8,
                 split="train_predictive", develop=False):
        self.dataset_location = dataset_location
        self.features = features
        self.n_hypotheses = n_hypotheses
        self.n_references = n_references

        self.max_dataset_size = max_dataset_size  # Amount of rows
        self.split = split
        self.develop = develop

        self.current_dataset_size = 0
        self.current_dataset_number = 0

        self.total_count = 0

        self.columns = [
                           "source", 'target', 'hypotheses',

                       ] + features

        self.current_data = None
        self.init_current_data()

        self.indices_count = {
            "id": [], "count": []
        }

        self.path_manager = get_path_manager()

        # Keep track of the ids (for the metadata)

        self.dataset_indices = []
        self.current_start_id = 0
        self.current_end_id = 0

        self.last_saved = 0

        self.create_dirs()

    def init_current_data(self):
        # Create dict that will hold the data.
        self.current_data = {**{col: [] for col in self.columns}, "id": []}

        self.current_data["utilities"] = []

        self.current_start_id = self.total_count
        self.current_dataset_size = 0

    def get_dataset_location(self):

        location = self.dataset_location + '{}_{}/{}/'.format(self.n_hypotheses, self.n_references, self.split)
        if self.develop:
            location += 'develop/'

        return location

    def add_rows(self, rows):
        '''
        Add rows to the dataset
        Saves intermediate if needed
        :param rows:
        :return:
        '''

        size = len(rows[self.columns[0]])

        ids = [self.total_count + i for i in range(size)]

        for col in self.columns:
            self.current_data[col] += rows[col]

        utilities = []
        for util, counts in zip(rows["utilities"], rows["utilities_count"]):
            temp = []
            for u, c in zip(util, counts):
                temp += [u] * c
            utilities.append(temp)

        self.current_data["utilities"] += utilities

        self.current_data["id"] += ids

        self.indices_count["id"] += ids
        self.indices_count["count"] += rows["count"]

        self.total_count += size
        self.current_dataset_size += size

        if self.current_dataset_size >= self.max_dataset_size:
            self.current_end_id = self.total_count
            self.save_intermediate()
            self.init_current_data()

            self.current_dataset_number += 1

    def finalize(self):
        '''
        Finalizes the whole process (saves the metadata)
        :return:
        '''
        print("finalizing")
        # Save last time if needed:
        if self.last_saved < self.total_count:
            self.current_end_id = self.total_count
            self.save_intermediate()
            self.current_dataset_number += 1

        # Save metadata as json
        self.save_metadata()
        # Save the indices dataset
        self.save_indices()

    def save_metadata(self):

        metadata = {
            "dataset_indices": self.dataset_indices,
            "size": self.total_count,
            "n_dataset": self.current_dataset_number
        }
        # Save as json
        save_file = '{}/metadata.json'.format(self.get_dataset_location())
        path = self.path_manager.get_abs_path(save_file)
        save_dict_as_json(metadata, path)

    def save_indices(self):
        save_file = '{}/indices_count.csv'.format(self.get_dataset_location())
        path = self.path_manager.get_abs_path(save_file)

        save_dict_as_csv(self.indices_count, path)

    def save_intermediate(self):
        '''
        Saves the intermediate
        :return:
        '''
        print("save_intermediate")
        save_path = self.get_save_path()

        #
        # Save as arrow dataset

        dataset = Dataset.from_dict(self.current_data)

        table = dataset.data.table
        save_arrow_file(table, save_path)

        # Add the dataset indices (for the metadata)
        self.dataset_indices.append((self.current_start_id, self.current_end_id))

        self.last_saved = self.current_end_id

    def create_dirs(self):
        # Create the dirs if not exists
        dataset_path = self.path_manager.get_abs_path(self.get_dataset_location())
        Path(dataset_path).mkdir(parents=True, exist_ok=True)
        dataset_path = self.path_manager.get_abs_path(self.get_dataset_location() + 'data/')
        Path(dataset_path).mkdir(parents=True, exist_ok=True)

    def get_save_path(self):

        file_name = '{}/data/{}.arrow'.format(self.get_dataset_location(), self.current_dataset_number)

        path = self.path_manager.get_abs_path(file_name)

        return path

# class PreBayesDatasetCreator:
#
#     def __init__(self, dataset_location, features, n_hypotheses=10, n_references=100, max_dataset_size=4096 * 8, split="train_predictive", develop=False):
#         self.dataset_location = dataset_location
#         self.features = features
#         self.n_hypotheses = n_hypotheses
#         self.n_references = n_references
#
#
#
#         self.max_dataset_size = max_dataset_size  # Amount of rows
#         self.split = split
#         self.develop = develop
#
#         self.current_dataset_size = 0
#         self.current_dataset_number = 0
#
#         self.total_count = 0
#
#         self.columns = [
#                            "source", 'target', 'hypotheses', 'utilities', 'utilities_count',
#
#                        ] + features
#
#         self.current_data = None
#         self.init_current_data()
#
#         self.indices_count = {
#             "id": [], "count": []
#         }
#
#         self.path_manager = get_path_manager()
#
#         # Keep track of the ids (for the metadata)
#
#         self.dataset_indices = []
#         self.current_start_id = 0
#         self.current_end_id = 0
#
#         self.last_saved = 0
#
#         self.create_dirs()
#
#     def init_current_data(self):
#         # Create dict that will hold the data.
#         self.current_data = {**{col: [] for col in self.columns}, "id": []}
#         self.current_start_id = self.total_count
#         self.current_dataset_size = 0
#
#     def get_dataset_location(self):
#
#         location = self.dataset_location + '{}_{}/{}/'.format(self.n_hypotheses, self.n_references, self.split)
#         if self.develop:
#             location += 'develop/'
#
#         return location
#
#     def add_rows(self, rows):
#         '''
#         Add rows to the dataset
#         Saves intermediate if needed
#         :param rows:
#         :return:
#         '''
#
#         size = len(rows[self.columns[0]])
#
#         ids = [self.total_count + i for i in range(size)]
#
#         for col in self.columns:
#             self.current_data[col] += rows[col]
#
#         self.current_data["id"] += ids
#
#         self.indices_count["id"] += ids
#         self.indices_count["count"] += rows["count"]
#
#         self.total_count += size
#         self.current_dataset_size += size
#
#         if self.current_dataset_size >= self.max_dataset_size:
#             self.current_end_id = self.total_count
#             self.save_intermediate()
#             self.init_current_data()
#
#             self.current_dataset_number += 1
#
#     def finalize(self):
#         '''
#         Finalizes the whole process (saves the metadata)
#         :return:
#         '''
#         print("finalizing")
#         # Save last time if needed:
#         if self.last_saved < self.total_count:
#             self.current_end_id = self.total_count
#             self.save_intermediate()
#             self.current_dataset_number += 1
#
#         # Save metadata as json
#         self.save_metadata()
#         # Save the indices dataset
#         self.save_indices()
#
#     def save_metadata(self):
#
#         metadata = {
#             "dataset_indices": self.dataset_indices,
#             "size": self.total_count,
#             "n_dataset": self.current_dataset_number
#         }
#         # Save as json
#         save_file = '{}/metadata.json'.format(self.get_dataset_location())
#         path = self.path_manager.get_abs_path(save_file)
#         save_dict_as_json(metadata, path)
#
#     def save_indices(self):
#         save_file = '{}/indices_count.csv'.format(self.get_dataset_location())
#         path = self.path_manager.get_abs_path(save_file)
#
#         save_dict_as_csv(self.indices_count, path)
#
#     def save_intermediate(self):
#         '''
#         Saves the intermediate
#         :return:
#         '''
#         print("save_intermediate")
#         save_path = self.get_save_path()
#
#         #
#         # Save as arrow dataset
#
#         dataset = Dataset.from_dict(self.current_data)
#
#         table = dataset.data.table
#         save_arrow_file(table, save_path)
#
#         # Add the dataset indices (for the metadata)
#         self.dataset_indices.append((self.current_start_id, self.current_end_id))
#
#         self.last_saved = self.current_end_id
#
#     def create_dirs(self):
#         # Create the dirs if not exists
#         dataset_path = self.path_manager.get_abs_path(self.get_dataset_location())
#         Path(dataset_path).mkdir(parents=True, exist_ok=True)
#         dataset_path = self.path_manager.get_abs_path(self.get_dataset_location() + 'data/')
#         Path(dataset_path).mkdir(parents=True, exist_ok=True)
#
#     def get_save_path(self):
#
#         file_name = '{}/data/{}.arrow'.format(self.get_dataset_location(), self.current_dataset_number)
#
#         path = self.path_manager.get_abs_path(file_name)
#
#         return path
