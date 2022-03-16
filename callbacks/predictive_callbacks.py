import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.callbacks import Callback


class RankCheckCallback(Callback):

    def __init__(self, sources, list_of_targets):
        '''
        This callback checks how well the model learned to rank given targets
        For each source we calculate the ranks of each sentence and check if they match the real rank.
        The distance metric that is used is the  in which 0 is a perfect score and the higher, the worse it is
        :param sources: the sources to translate
        :param targets: list of targets for each source should be from best to worse.
        '''
        super().__init__()
        self.sources = sources
        self.list_of_targets = list_of_targets

    def on_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        distances = []

        for source, targets in zip(self.sources, self.list_of_targets):
            sources = [source] * len(targets)
            avgs, stds = pl_module.predict(sources, targets)

            avgs = avgs.cpu().numpy().flatten()

            # Real rank
            real_rank = np.arange(len(targets))

            # Given rank:
            order = avgs.argsort()

            predicted_rank = order.argsort()

            distance = self.calc_distance(predicted_rank, real_rank)
            distances.append(distance)

    def calc_distance(self, predicted_rank, real_rank):
        return sum(predicted_rank == real_rank)


class MyShuffleCallback(Callback):
    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

    def on_train_epoch_start(self, trainer, pl_module):
        self.dataset.shuffle()


