import os
from pathlib import Path

import pandas as pd

import ast


from os.path import exists

from datasets import Dataset
from pandas import RangeIndex

from custom_datasets.PreprocessedCachedBayesRiskDataset import PreprocessedCachedBayesRiskDataset
from custom_datasets.utils import save_arrow_file, load_arrow_file

'''
Code for loading bayes risk datasets.

The format of the datasets are:
{split}_predictive_{sample_method}_scores_{n_hypothesis}_{n_references}
'''


class PreprocessedBayesRiskDatasetLoader:
    def __init__(self, config, pl_model=None, develop=False, preprocess_batch_size=16, home_dir=None):
        self.splits = [
            "train_predictive",
            "validation_predictive"
        ]

        self.preprocess_dir = config["preprocess_dir"]
        self.base_dir = config["base_dir"]

        self.sampling_method = config["sampling_method"]

        self.n_hypothesis = config["n_hypothesis"]
        self.n_references = config["n_references"]

        self.pl_model = pl_model

        self.develop = develop
        self.preprocess_batch_size = preprocess_batch_size
        if home_dir:
            self.home_dir = home_dir
        else:
            self.home_dir = str(Path.home())

    def load(self):

        # First get the dataset file and check if

        result = {

        }
        file_names = {}
        preprocessed = {

        }

        for split in self.splits:

            file_names[split] = self.get_dataset_path(self.preprocess_dir, split, preprocessed=True)

            if not exists(file_names[split]):
                file_names[split] = self.get_dataset_path(self.base_dir, split, preprocessed=False)
                preprocessed[split] = False
            else:
                preprocessed[split] = True

        # Loads the dataset
        for split in self.splits:

            # If we can use the pr
            if preprocessed[split]:
                dataset, ref_table = self.load_preprocessed_dataset(split=split)

                result['{}_dataset'.format(split)] = dataset
                result['{}_ref_table'.format(split)] = ref_table

            else:
                save_location = self.get_dataset_path(self.preprocess_dir, split, preprocessed=True)

                dataset, ref_table = self.load_and_preprocess(split,
                                                              save=True,
                                                              save_location=save_location)
                result['{}_dataset'.format(split)] = dataset
                result['{}_ref_table'.format(split)] = ref_table

        final_results = {}
        for split in self.splits:
            final_results[split] = PreprocessedCachedBayesRiskDataset(result['{}_dataset'.format(split)],
                                                                      result['{}_ref_table'.format(split)], split,
                                                                      self.pl_model.feature_names)
        return final_results

    def get_dataset_path(self, dir, split, preprocessed=False, ):

        dataset_file = "{}_{}_scores_{}_{}".format(split, self.sampling_method, self.n_hypothesis,
                                                   self.n_references)

        if self.develop:
            dataset_file += '_develop'

        if preprocessed:
            dataset_file += '_preprocessed.arrow'
        else:
            dataset_file += '.csv'
        dataset_path = os.path.join(self.home_dir, dir, dataset_file)
        return dataset_path

    def load_and_preprocess(self, split, save_location='', save=False, batch_size=16):

        dataset = self.load_split(split, )

        return self.preprocess(dataset, save=save, save_location=save_location, batch_size=batch_size)

    def load_split(self, split, ):
        dataset_file = self.get_dataset_path(self.base_dir, split=split,
                                             preprocessed=False)

        dataset = pd.read_csv(dataset_file, sep="\t")

        #
        dataset["hypothesis"] = dataset["hypothesis"].map(lambda x: ast.literal_eval(x))

        dataset["utilities"] = dataset["utilities"].map(lambda x: ast.literal_eval(x))
        dataset["count"] = dataset["count"].map(lambda x: ast.literal_eval(x))

        # Explode the dataset
        dataset = dataset.explode(["hypothesis", "utilities", "count"], ignore_index=True)

        # Each dict must be a string again for some reason
        dataset["utilities"] = dataset["utilities"].map(lambda x: str(x))
        dataset = Dataset.from_pandas(dataset)

        return dataset

    def preprocess(self, dataset, save=False, save_location='', batch_size=16):
        # Preprocess the dataset

        # We want to make sure that the utilities are easy to process
        def util_process(batch):

            temp = [ast.literal_eval(u) for u in batch["utilities"]]
            # To make sure we have the right values together
            u_c = []  # [ (k,v)  for u in temp for k,v in u.items()]

            for u in temp:
                u_c.append([(k, v) for k, v in u.items()])

            utilities = []
            for u in u_c:
                utilities.append([k for (k, v) in u])
            counts = []
            for u in u_c:
                counts.append([v for (k, v) in u])

            return {'utilities': utilities, 'utilities_count': counts}

        ref_dataset = dataset.map(util_process, batched=True, )

        ref_dataset = ref_dataset.map(self.pl_model.preprocess_function, batched=True, batch_size=batch_size)

        ref_table = ref_dataset.data.table
        # Save the result
        if save:
            save_arrow_file(ref_table, save_location)
            # pq.write_table(ref_table, save_location)

        # Create the dataset without all the features
        no_features_dataset = ref_dataset.remove_columns(
            self.pl_model.feature_names + ["sources", "hypothesis", "utilities", "utilities_count"])

        df = no_features_dataset.to_pandas()
        df['ref_id'] = RangeIndex(start=0, stop=df.index.stop, step=1)
        df = df.reindex(df.index.repeat(df["count"]))
        df = df.reset_index(level=0, inplace=False, drop=True)
        dataset = Dataset.from_pandas(df)
        return dataset, ref_table

    def load_preprocessed_dataset(self, split):
        dataset_file = self.get_dataset_path(self.preprocess_dir, split=split, preprocessed=True)

        # Load the parquet file

        ref_table = load_arrow_file(dataset_file)  # pq.read_table(dataset_file, )

        dataset = Dataset(ref_table)
        no_features_dataset = dataset.remove_columns(self.pl_model.feature_names + ["sources", "hypothesis"])

        df = no_features_dataset.to_pandas()
        df['ref_id'] = RangeIndex(start=0, stop=df.index.stop, step=1)
        df = df.reindex(df.index.repeat(df["count"]))
        df = df.reset_index(level=0, inplace=False, drop=True)
        dataset = Dataset.from_pandas(df)
        return dataset, ref_table
