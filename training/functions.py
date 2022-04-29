
# File to train a model from scratch


import torch
import yaml
import numpy as np
import pytorch_lightning as pl
from datasets import Dataset
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers

from custom_datasets.BayesRiskDataset.BayesRiskDatasetLoader import BayesRiskDatasetLoader


from models.estimation.ReferenceModels.factory import ReferenceModelFactory



class Collate:

    def __init__(self, n_references=3):
        self.n_references = n_references

    def __call__(self, batch):
        sources = [b["source"] for b in batch]
        hypotheses = [b["hypotheses"] for b in batch]


        # Get rando mindices
        references_indices = [np.random.choice(len(b["references"]), size=self.n_references, p=b["p_references"], replace=True).tolist() for b
                      in batch]



        # Pick according to the index
        references = np.array([np.array(b["references"])[i] for b, i in zip(batch, references_indices)])


        ref_ids = np.array([np.array(b["ref_ids"])[i] for b, i in zip(batch, references_indices)])





        #references = np.stack(references).flatten() # This can be used to do more parellel compuytation

        references = np.swapaxes(references, 0,1).tolist()
        ref_ids = np.swapaxes(ref_ids, 0,1).tolist()



        score = torch.tensor([b["score"] for b in batch])

        hyp_ids = [b["hypotheses_ids"] for b in batch]


        return sources, hypotheses, references, score, hyp_ids, ref_ids


def calc_score(x):
    utils = np.array(x["utilities"])
    utilities_count = np.array(x["utilities_count"])
    score = float(np.sum(utils * utilities_count) / np.sum(utilities_count))
    return score


def calc_p_references(x):
    count = np.array(x["count"])
    p = count / np.sum(count)
    return p

def create_ids(x):
    id = x.name


    ids = ["{}_{}".format(id, i) for i in range(len(x["hypotheses"]))]
    return ids



def preprocess_dataset(data):
    # First we add the best to the list, together with it's utility.
    data["references"] = data["hypotheses"]


    data["p_references"] = data.apply(calc_p_references, axis=1)
    data["hypotheses_ids"] = data.apply(create_ids, axis=1)
    data["ref_ids"] = data["hypotheses_ids"]

    df_exploded = data.explode(["hypotheses", "utilities", "count", "hypotheses_ids"], ignore_index=True)

    df_exploded["score"] = df_exploded.apply(calc_score, axis=1)

    return df_exploded




def train_estimation_model(config_ref, smoke_test=False):
    pl.seed_everything(1)
    np.random.seed(1)

    with open(config_ref, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    factory = ReferenceModelFactory(config["model_config"])

    model = factory.create_model()

    dataset_config = config["dataset"]

    n_hypotheses = dataset_config["n_hypotheses"]
    n_references = dataset_config["n_references"]
    utility = dataset_config["utility"]
    sampling_method = dataset_config["sampling_method"]

    dataset_dir = dataset_config["dir"]
    train_dataset_loader = BayesRiskDatasetLoader("train_predictive", n_hypotheses, n_references,
                                                  sampling_method, utility, develop=False,
                                                  base=dataset_dir)

    validation_dataset_loader = BayesRiskDatasetLoader("validation_predictive", n_hypotheses, n_references,
                                                       sampling_method, utility, develop=False,
                                                       base=dataset_dir)

    if smoke_test:
        train_dataset = train_dataset_loader.load(type="pandas").data.iloc[:256].copy()
        validation_dataset = validation_dataset_loader.load(type="pandas").data.iloc[:256].copy()

    else:
        train_dataset = train_dataset_loader.load(type="pandas").data
        validation_dataset = validation_dataset_loader.load(type="pandas").data

    # We want to transform this thing to a extended dataset. We use pandas for this
    validation_dataset = preprocess_dataset(validation_dataset)
    train_dataset = preprocess_dataset(train_dataset)

    train_dataset = Dataset.from_pandas(train_dataset)

    validation_dataset = Dataset.from_pandas(validation_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=Collate(),
                                  batch_size=config["batch_size"], shuffle=True, )
    val_dataloader = DataLoader(validation_dataset,
                                collate_fn=Collate(),
                                batch_size=config["batch_size"], shuffle=False, )


    # Check if training loss keeps improving (if not we can surely stop)
    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.00, patience=10, verbose=False, mode="min",
                                        check_finite=True,
                                        divergence_threshold=3)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=config["log_dir"])


    max_epochs = 1 if smoke_test else config["max_epochs"]
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=1,
        progress_bar_refresh_rate=1,
        val_check_interval=0.5,
        callbacks=[LearningRateMonitor(logging_interval="step"), early_stop_callback],
        logger=tb_logger,
        accumulate_grad_batches=config["accumulate_grad_batches"]
    )

    # create the dataloaders
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, )


