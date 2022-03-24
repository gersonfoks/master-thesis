import os
import shutil

from pytorch_lightning import Callback


class CheckpointCallback(Callback):
    def __init__(self, factory, path, metric="val_loss", top_k=3):
        super().__init__()

        self.factory = factory

        self.path = path

        self.model_loss = []

        self.counter = 0
        self.metric = metric

        self.top_k = top_k

    def on_validation_end(self, trainer, pl_module):

        path = self.path + str(self.counter) + '/'

        if self.metric in trainer.logged_metrics.keys():

            self.factory.save(pl_module, path)

            self.model_loss.append((trainer.logged_metrics[self.metric].cpu().item(), path))

            if len(self.model_loss) > self.top_k:
                self.model_loss.sort(key=lambda x: x[0], reverse=False)

                to_delete = self.model_loss[self.top_k:]

                for (val, path) in to_delete:

                    if os.path.exists(path):
                        shutil.rmtree(path)

        self.counter += 1
