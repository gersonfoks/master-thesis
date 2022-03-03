# File to train a model from scratch

import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import ast




import numpy as np





def count_to_mean(counter):
    total_count = 0
    total = 0
    for value, c in counter.items():
        total += value * c
        total_count += c

    return total / total_count





def get_ref_collate_fn(ref_table, pl_model):
    '''
    Returns a collate function that queries the ref_table for the features.
    :param ref_table:
    :param pl_model:
    :return:
    '''
    def collate_fn(batch):
        # First get the features
        ids = [s["ref_id"] for s in batch]

        utilities = torch.tensor([count_to_mean(ast.literal_eval(s["utilities"])) for s in batch])

        info = ref_table.take(ids)
        sources = info["sources"].flatten()
        hypothesis = info["hypothesis"].flatten()
        features = {feature_name: torch.Tensor(np.stack(info[feature_name].to_numpy())) for feature_name in
                    pl_model.feature_names}

        return features, (sources, hypothesis), utilities

    return collate_fn


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str, default='./trained_models/predictive/tatoeba-de-en-predictive.yml',
                        help='config to load model from')

    parser.add_argument('--develop', dest='develop', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.set_defaults(develop=False)

    args = parser.parse_args()

    parsed_config = parse_predictive_config(args.config, pretrained=False, develop=args.develop, preprocessing=True)
    nmt_model = parsed_config["nmt_model"]
    nmt_model.eval()
    pl_model = parsed_config["pl_model"]
    tokenizer = parsed_config["tokenizer"]
    pl_model.set_mode("features")
    train_dataset = parsed_config["datasets"]["train_dataset"]
    val_dataset = parsed_config["datasets"]["validation_dataset"]
    train_ref_table = parsed_config["datasets"]["train_ref_table"]
    validation_ref_table = parsed_config["datasets"]["validation_ref_table"]
    #
    # train_dataset = train_dataset.map(pl_model.preprocess_function, batched=True, batch_size=16)
    # val_dataset = val_dataset.map(pl_model.preprocess_function, batched=True, batch_size=16)
    #
    # train_dataset, train_ref_table = preprocess(train_dataset, pl_model)
    # val_dataset, val_ref_table = preprocess(val_dataset, pl_model)


    train_collate_fn = get_ref_collate_fn(train_ref_table, pl_model)
    val_collate_fn = get_ref_collate_fn(validation_ref_table, pl_model)

    train_dataloader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=2048, shuffle=True, )
    val_dataloader = DataLoader(val_dataset, collate_fn=val_collate_fn, batch_size=2048, shuffle=False, )

    trainer = pl.Trainer(gpus=1, accumulate_grad_batches=1, max_epochs=25, val_check_interval=0.5)  #

    trainer.fit(pl_model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, )

    # save_model(pl_model, parsed_config["config"], "./data/develop_model")
    #
    # loaded_pl_model = load_model("./data/develop_model", type="MSE")

    # Validate new model (just to be sure)
    # trainer.validate(loaded_pl_model, val_dataloaders=val_dataloader, )


if __name__ == '__main__':
    main()
