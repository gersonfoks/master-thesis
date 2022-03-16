from torch.utils.data import Dataset

import pandas as pd


class SampleDataset(Dataset):
    '''
    This dataset is pretty simply a wrapper around a dictionairy.
    We can implement a more complex structure if needed for speed purposes
    '''

    def __init__(self, data=None, split='train_predictive', n_samples=10):
        super().__init__()

        self.columns = [
            "source", 'target', 'samples', "count"
        ]

        if data != None:
            self.data = data
        else:
            self.data = {col: [] for col in self.columns}

        self.split = split
        self.n_samples = n_samples

    def __len__(self):
        return len(self.data['source'])

    def add_samples(self, source, target, samples_counter):
        self.data['source'].append(source)
        self.data['target'].append(target)

        samples = []
        counts = []
        for sample, count in samples_counter.items():
            samples.append(sample)
            counts.append(count)
        self.data["samples"].append(samples)
        self.data["count"].append(counts)

    def save(self, path):

        df = pd.DataFrame.from_dict(self.data)

        df.to_csv(path, index=False, sep="\t")

    def __iter__(self):
        # Bit of a hack but works for now
        df = pd.DataFrame.from_dict(self.data)
        return df.iterrows()