import math
import pickle
import numpy as np
from datasets import load_dataset

# Load the dataset

seed = 1
# First is for finetuning/train NMT model, second is for train the predictive model, third is for validation, rest is for test.
dataset = load_dataset("tatoeba", lang1="de", lang2="en", )["train"]

n_samples = len(dataset)
print(n_samples)
train_MLE_size = int(0.9 * n_samples)
evaluation_MLE_size = 2500
evaluation_predictive_size = 2500
test_size = 5000
train_predictive_size = n_samples - train_MLE_size - evaluation_predictive_size - evaluation_MLE_size - test_size

print(train_MLE_size)
print(train_predictive_size)

indices = np.arange(stop=len(dataset))
np.random.seed(seed)
np.random.shuffle(indices)

train_MLE_end = train_MLE_size
train_predictive_end = train_MLE_end + train_predictive_size
validation_MLE_end = evaluation_MLE_size + train_predictive_end
evaluation_predictive_end = evaluation_predictive_size + validation_MLE_end

train_MLE_indices = indices[:train_MLE_end]
train_predictive_indices = indices[train_MLE_end:train_predictive_end]
validation_MLE_indices = indices[train_predictive_end: validation_MLE_end]
validation_predictive_indices = indices[validation_MLE_end: evaluation_predictive_end]
test_indices = indices[evaluation_predictive_end:]

# safe the splits

splits = {
    "train_MLE": train_MLE_indices,
    "train_predictive": train_predictive_indices,
    "validation_MLE": validation_MLE_indices,
    "validation_predictive": validation_predictive_indices,
    "test": test_indices
}

with open("splits_tatoeba.pkl", "wb") as f:
    pickle.dump(splits, f)

with open("splits_tatoeba.pkl", "rb") as f:
    splits = pickle.load(f)
    print(splits)
