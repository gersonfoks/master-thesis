### Script for testing a model

from datasets import tqdm, load_metric, Dataset
import torch

from models.estimation.NMT_bayes_risk_model import NMTBayesRisk
from utils.config_utils import load_model
from utils.dataset_utils import get_dataset



def main():

    data = get_dataset("tatoeba", source="de",
                       target="en")

    test_data = Dataset.from_dict(data["test"][:100])
    print(type(test_data))
    loaded_pl_model = load_model("./data/develop_model").eval()
    model = NMTBayesRisk(loaded_pl_model)

    sacreblue_metric = load_metric('sacrebleu')
    with torch.no_grad():
        for x in tqdm(test_data):

            source = x["translation"]["de"]
            target = [[x["translation"]["en"]]]

            translation = model.forward(source, n_samples_per_source=96)

            print(translation)
            print(target)

            sacreblue_metric.add_batch(predictions=[translation], references=target)
        bleu = sacreblue_metric.compute()
        test_results = {

            "sacrebleu": bleu
        }

        print("results")
        print(test_results)




if __name__ == '__main__':
    main()
