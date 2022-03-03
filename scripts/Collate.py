from datetime import datetime

import torch
import ast
import numpy as np


def mean_util(utilities, count):
    return float(np.sum(utilities * count) / np.sum(count))

def random_util(utilities, count):
    '''
    Pick a random utlitity, chances are weighted by the count
    :param utilities:
    :param count:
    :return:
    '''
    raise NotImplementedError()


class Collator:

    def __init__(self, feature_names, util_func):

        self.feature_names = feature_names
        self.util_func = util_func

    def __call__(self, batch):
        utilities = [self.util_func(s["utilities"], s["utilities_count"]) for s in batch]

        utilities = torch.tensor(utilities)

        sources = [s["source"] for s in batch]
        hypotheses = [s["hypothesis"] for s in batch]

        features = {}

        for feature_name in self.feature_names:
            features[feature_name] = torch.stack([b[feature_name] for b in batch])

        return features, (sources, hypotheses), utilities


class QueryCollator:

    def __init__(self, ref_table, feature_names):
        self.ref_table = ref_table
        self.feature_names = feature_names

    def __call__(self, batch):
        ids = [s["ref_id"] for s in batch]

        info = self.ref_table.take(ids, )

        utitilities = torch.tensor([mean_util(ast.literal_eval(s["utilities"])) for s in batch])

        sources = info["sources"].flatten()
        hypotheses = info["hypothesis"].flatten()

        features = {feature_name: torch.Tensor(np.stack(info[feature_name].to_numpy())) for feature_name in
                    self.feature_names}

        return features, (sources, hypotheses), utitilities
