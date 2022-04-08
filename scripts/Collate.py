from datetime import datetime

import torch
import ast
import numpy as np




def mean_util(utilities, count):
    x = float(np.sum(utilities * count) / np.sum(count))**2


    return x


def random_util(utilities, count):
    '''
    Pick a random utlitity, chances are weighted by the count
    :param utilities:
    :param count:
    :return:
    '''
    p = count / np.sum(count)
    x = np.random.choice(utilities, p=p)

    return x


def full_util(utilities, count):
    r = []
    for u, c in zip(utilities, count):
        r += [u] * c


    return np.array(r)


util_functions = {
    'gaussian': lambda x: x,
    'gaussian-mixture': lambda x: x,
    'gaussian-full': lambda x: x,
    'student-t-mixture': lambda x: x,
    'MSE': lambda x: np.mean(x,axis=-1),
}


class Collator:

    def __init__(self, feature_names, util_func):
        self.feature_names = feature_names
        self.util_func = util_func

    def __call__(self, batch):

        utilities = np.stack([self.util_func(s["utilities"], s["utilities_count"]) for s in batch])

        utilities = torch.tensor(utilities)

        sources = [s["source"] for s in batch]
        hypotheses = [s["hypothesis"] for s in batch]

        features = {}

        for feature_name in self.feature_names:
            features[feature_name] = torch.Tensor(np.stack([b[feature_name] for b in batch]))

        return features, (sources, hypotheses), utilities


class SequenceCollator:

    def __init__(self, feature_names, util_func):
        self.feature_names = feature_names
        self.util_func = util_func

    def __call__(self, batch):
        # First get the features

        utilities = self.util_func(np.stack([s["utilities"] for s in batch]))

        #utilities = np.ones((len(batch), 100)) #np.stack([ np.array(s["utilities"]) for s in batch])
        utilities = torch.tensor(utilities, dtype=torch.float32)

        sources = [s["source"] for s in batch]
        hypotheses = [s["hypothesis"] for s in batch]



        features = {}
        for feature_name in self.feature_names:
            s = [b[feature_name] for b in batch]

            new_features = add_mask_to_features(feature_name, s)
            features = {**features, **new_features}


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



def add_mask_to_features(name, features):
    '''
    Adds attention and append features.
    :param name:
    :param features:
    :return:
    '''
    # TODO: maybe it is possible to do this vectorized (for speedup)

    mask_list = []
    new_features = []

    lengths = [t.shape[0] for t in features]

    max_length = np.max(lengths)


    for l, t in zip(lengths, features):
        to_add = max_length - l

        t = np.stack(t).astype(np.float32)

        if to_add > 0:
            shape = t.shape
            shape = list(shape)
            shape[0] = to_add

            zero_feature = np.zeros(shape, dtype=np.float32)

            new_feature = np.concatenate((t, zero_feature))
            mask = np.concatenate((np.zeros((l,)), np.ones((to_add,))))
        else:
            new_feature = t
            mask = np.zeros((l,))

        mask_list.append(mask)
        new_features.append(new_feature)


    r_attention = torch.Tensor(np.stack(mask_list))
    r_features = torch.Tensor(np.stack(new_features))


    return {name: r_features, '{}_mask'.format(name): r_attention}

