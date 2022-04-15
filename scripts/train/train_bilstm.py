'''
Temp Helper file
'''
import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from datasets import Dataset
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE, Unigram
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from custom_datasets.BayesRiskDataset.BayesRiskDatasetLoader import BayesRiskDatasetLoader
from custom_datasets.PreprocessedBayesRiskDataset.PreBayesRiskDatasetCreator import PreBayesDatasetCreator
from models.bilstm import BIListmModel, PlLSTMModel
from models.predictive.PLPredictiveModelFactory import PLPredictiveModelFactory



class Collate:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.encode("<PAD>").ids[0]
        #self.collator = DataCollatorWithPadding(tokenizer, padding=True, return_tensors="pt")

    def __call__(self, batch):
        model_input = [b["model_input"] for b in batch]

        score = torch.tensor([b["score"] for b in batch])
        # tokenize:

        ids = [enc.ids for enc in self.tokenizer.encode_batch(model_input)]
        lenghts = np.array([len(i) for i in ids])
        sorted_lenghts = np.argsort(lenghts)



        model_input_tokenized = [torch.tensor(i) for i in ids]


        #model_input_tokenized = [torch.tensor(enc.ids) for enc in self.tokenizer.encode_batch(model_input)]
        #model_input_tokenized = self.collator(model_input)


        #model_input = pad_sequence(model_input_tokenized, padding_value=self.pad_id, batch_first=True)


        #packed_model_input = pack_padded_sequence(model_input, lenghts, enforce_sorted=False,batch_first=True)
        packed_model_input = pack_sequence(model_input_tokenized, enforce_sorted=False)
        #print(packed_model_input)
        # print(model_inputpa

        return packed_model_input, score


def concat_source_and_hypotheses(x):
    source = x["source"]
    hypotheses = x["hypotheses"]

    return source + " <SEP> " + hypotheses


def calc_score(x):

    utils = np.array(x["utilities"])
    utilities_count = np.array(x["utilities_count"])
    score = float(np.sum(utils * utilities_count) / np.sum(utilities_count))
    return score

def preprocess_dataset(data):
    # First we add the best to the list, together with it's utility.

    df_exploded = data.explode(["hypotheses", "utilities", "count"], ignore_index=True)




    df_exploded["model_input"] = df_exploded.apply(concat_source_and_hypotheses, axis=1)

    df_exploded["score"] = df_exploded.apply(calc_score, axis=1)

    return df_exploded



unk_token = "<UNK>"  # token for unknown words
spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>", "<PAD>"]  # special tokens




def main():
    parser = argparse.ArgumentParser(description='Preprocesses a dataset')


    parser.add_argument('--develop', dest='develop', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.set_defaults(develop=False)

    parser.add_argument('--dataset-dir', type=str, default='predictive/tatoeba-de-en/data/raw/')

    parser.add_argument('--utility', type=str, default="unigram-f1")

    parser.add_argument('--save-dir', type=str, default='predictive/tatoeba-de-en/data/preprocessed/')

    parser.add_argument('--n-hypotheses', type=int, default=10, help='Number of hypothesis to use')
    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    parser.add_argument('--n-references', type=int, default=100, help='Number of references for each hypothesis')


    args = parser.parse_args()

    dataset_dir = args.dataset_dir


    train_dataset_loader = BayesRiskDatasetLoader("train_predictive", args.n_hypotheses, args.n_references,
                                                       args.sampling_method, args.utility, develop=args.develop, base=dataset_dir)

    validation_dataset_loader = BayesRiskDatasetLoader("validation_predictive", args.n_hypotheses, args.n_references,
                                              args.sampling_method, args.utility, develop=args.develop,
                                              base=dataset_dir)

    train_dataset = train_dataset_loader.load(type="pandas").data
    validation_dataset = validation_dataset_loader.load(type="pandas").data

    # We want to transform this thing to a extended dataset. We use pandas for this
    validation_dataset = preprocess_dataset(validation_dataset)
    train_dataset = preprocess_dataset(train_dataset)


    # Next we need to create a tokenizer

    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    trainer = BpeTrainer(special_tokens=spl_tokens)

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = normalizers.NFKC()

    data_to_tokenize = train_dataset["model_input"].iloc[:]


    tokenizer.train_from_iterator(data_to_tokenize, trainer)


    train_dataset = Dataset.from_pandas(train_dataset)

    validation_dataset = Dataset.from_pandas(validation_dataset)

    # Next we can start iterating trough the dataset

    collate_fn = Collate(tokenizer)
    train_data_loader = DataLoader(train_dataset, batch_size=2048, collate_fn=collate_fn, shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=512, collate_fn=collate_fn, shuffle=False)


    model = BIListmModel(tokenizer.get_vocab_size())

    pl_model = PlLSTMModel(model).to("cuda")

    trainer = pl.Trainer(
        max_epochs=100,
        gpus=1,
        progress_bar_refresh_rate=1,
        val_check_interval=0.5,
        gradient_clip_val=5.0,
        callbacks=[LearningRateMonitor(logging_interval="epoch")]

    )

    # create the dataloaders
    trainer.fit(pl_model, train_dataloader=train_data_loader, val_dataloaders=validation_data_loader, )





if __name__ == '__main__':
    main()
