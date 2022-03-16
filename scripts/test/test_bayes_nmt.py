### Script for testing a model
import argparse

from datasets import tqdm, load_metric, Dataset
import torch

from models.MBR_model.MBRModel import MBRModel

from models.pl_predictive.PLPredictiveModelFactory import PLPredictiveModelFactory

from utils.dataset_utils import get_dataset



def main():
    parser = argparse.ArgumentParser(description='Test NMT model with MBR with preselected samples')
    parser.add_argument('--path', type=str, default='./',
                        help='path to load the model from')
    args = parser.parse_args()

    data = get_dataset("tatoeba", source="de",
                       target="en")

    test_data = Dataset.from_dict(data["test"][:100])
    pl_model, factory = PLPredictiveModelFactory.load(args.path)
    pl_model.set_mode("text")
    pl_model.eval()
    model = MBRModel(pl_model)

    sacreblue_metric = load_metric('sacrebleu')
    with torch.no_grad():
        for x in tqdm(test_data):

            source = x["translation"]["de"]
            target = [[x["translation"]["en"]]]
            print("target: ", target)
            translation = model.forward(source, n_samples_per_source=100)



            sacreblue_metric.add_batch(predictions=[translation], references=target)
        bleu = sacreblue_metric.compute()
        test_results = {

            "sacrebleu": bleu
        }

        print("results")
        print(test_results)




if __name__ == '__main__':
    main()
