from torch.utils.data import Dataset
import pandas as pd


class BayesRiskDataset(Dataset):

    def __init__(self, data, split, n_hypotheses=10):
        super().__init__()

        self.columns = [
            "source", 'target', 'hypotheses', 'utilities', 'utilities_count', "count"

        ]

        self.n_hypotheses = 10

        if data != None:
            self.data = data
        else:
            self.data = {col: [] for col in self.columns}

        self.split = split

    def __len__(self):
        return len(self.data["source"])

    def add_row(self, source, target, hypotheses, utilities, utilities_count, count):
        self.data['source'].append(source)
        self.data['target'].append(target)
        self.data['hypotheses'].append(hypotheses)
        self.data["utilities"].append(utilities)
        self.data["utilities_count"].append(utilities_count)

        self.data['count'].append(count)

    def __getitem__(self, item):
        return {col: self.data[col][item] for col in self.columns}
