### Script for generating the samples from the model
### These samples can then be used to generate stastics.

import argparse
import torch
from transformers import MarianTokenizer, AutoModelForSeq2SeqLM

from utils.dataset_utils import get_dataset, save_samples, load_samples
from utils.translation_model_utils import translate


def create_samples(model, tokenizer, dataset, method="ancestral"):
    source_texts = [txt["en"] for txt in dataset]
    return translate(model, tokenizer, source_texts, method=method)


def main():
    # Parse all the arguments
    parser = argparse.ArgumentParser(description='Writing the important statistics to a file')

    parser.add_argument('--model', type=str, default='Helsinki-NLP/opus-mt-de-en')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dataset', type=str, default="tatoeba", help="The dataset to create the statistics for")
    parser.add_argument('--n_samples', type=int, default=500, help="How many samples are used")

    parser.add_argument('--method', type=str, default="ancestral", help="method of sampling from the model")
    parser.add_argument("--name", type=str, default="test", help="The name used for saving the samples")

    args = parser.parse_args()

    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the model
    tokenizer = MarianTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)

    # model.eval()

    # Load the dataset
    splits = get_dataset(args.dataset)

    # Get the relevant dataset out
    dataset = splits["test"]["translation"][:args.n_samples]
    # print(dataset)
    # Create samples
    samples = create_samples(model, tokenizer, dataset, args.method)

    # Save the samples
    ref = "./data/samples_{}_{}.pkl".format(args.name, args.method)
    save_samples(samples, ref)

    print("Small sample:")
    print(samples[:10])


if __name__ == '__main__':
    main()
