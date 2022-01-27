import pickle

from datasets import load_dataset


def get_dataset(dataset_name):
    result = None
    if dataset_name == "tatoeba":
        dataset = load_dataset("tatoeba", lang1="de", lang2="en", )["train"]

        # Load the splits
        splits = load_pickle("./data/splits.pkl")
        result = {k: dataset[v] for k, v in splits.items()}
    else:
        raise ValueError("Not a known dataset: {}".format(dataset_name))

    return result


def save_pickle(to_pickle, ref):
    with open(ref, "wb") as f:
        pickle.dump(to_pickle, f)


def load_pickle(ref):
    r = None
    with open(ref, "rb") as f:
        r = pickle.load(f)
    return r


def save_samples(samples, ref):
    with open("{}".format(ref), "wb") as f:
        pickle.dump(samples, f)


def load_samples(ref):
    return load_pickle(ref)
