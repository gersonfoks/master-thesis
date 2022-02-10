import json
import pickle
import torch
from datasets import load_dataset, Dataset
from transformers import DataCollatorForSeq2Seq


def get_dataset(dataset_name, source='de', target='en'):
    result = None
    if dataset_name == "tatoeba":
        dataset = load_dataset("tatoeba", lang1=source, lang2=target, )["train"]

        # Load the splits
        splits = load_pickle("./data/splits_tatoeba.pkl")
        result = {k: Dataset.from_dict(dataset[v]) for k, v in splits.items()}
        print([len(v) for k, v in result.items()])
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


def save_dict_to_json(dict, ref):
    with open(ref, 'w') as fp:
        json.dump(dict, fp, )


def get_collate_fn(model, tokenizer, source, target):
    data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer,
                                           padding=True, return_tensors="pt")

    keys = [
        "input_ids",
        "attention_mask",
        "labels"
    ]

    def collate_fn(batch):
        new_batch = [{k: s[k] for k in keys} for s in batch]
        x_new = data_collator(new_batch)

        sources = [s["translation"][source] for s in batch]
        targets = [s["translation"][target] for s in batch]

        return x_new, (sources, targets)

    return collate_fn


def get_predictive_collate_fn(model, tokenizer, ):
    data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer,
                                           padding=True, return_tensors="pt")

    keys = [
        "input_ids",
        "attention_mask",
        "labels"
    ]

    def collate_fn(batch):
        new_batch = [{k: s[k] for k in keys} for s in batch]
        x_new = data_collator(new_batch)

        sources = [s["source"] for s in batch]
        hypothesis = [s["hypothesis"] for s in batch]

        # Group the averages and the standard deviations
        utilities = torch.Tensor([s["utility"] for s in batch])

        return x_new, (sources, hypothesis), utilities

    return collate_fn
