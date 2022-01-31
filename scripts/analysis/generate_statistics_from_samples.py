### Scripts for generating statistics from samples
# We use the same tokenizer for each sample to get consistent tokenization

import argparse
from collections import Counter
from transformers import MarianTokenizer
from utils.dataset_utils import get_dataset, load_samples, save_pickle, load_pickle


def get_bigrams(ids):
    bigrams = []
    for first, second in zip(ids[:-1], ids[1:]):
        bigrams.append((first, second))
    return bigrams


def get_skip_bigrams(ids):
    skip_bigrams = []
    for first, second in zip(ids[:-2], ids[2:]):
        skip_bigrams.append((first, second))
    return skip_bigrams


def create_n_grams_count(samples_ids):
    unigram_counts = Counter()
    bigram_counts = Counter()
    skip_bigram_counts = Counter()
    for ids in samples_ids:
        bi_grams = get_bigrams(ids)
        skip_bigrams = get_skip_bigrams(ids)

        unigram_counts.update(ids)
        bigram_counts.update(bi_grams)
        skip_bigram_counts.update(skip_bigrams)

    return {"unigrams": unigram_counts, "bigrams": bigram_counts, "skip_bigram": skip_bigram_counts}


def main():
    # Parse all the arguments
    parser = argparse.ArgumentParser(description='Writing the important statistics to a file of the given samples')
    parser.add_argument('--refs_samples', '--list', nargs='+', help='List of references to the samples', required=True)

    parser.add_argument('--model', type=str, default='Helsinki-NLP/opus-mt-de-en')

    parser.add_argument('--dataset', type=str, default="tatoeba", help="The dataset to create the statistics for")
    parser.add_argument('--n_samples', type=int, default=500, help="How many samples are used")

    parser.add_argument("--name", type=str, default="test", help="The name used for saving the statistics")
    args = parser.parse_args()

    # Load the tokenizer
    tokenizer = MarianTokenizer.from_pretrained(args.model)

    # Load the samples
    list_of_samples = [load_samples(ref) for ref in args.refs_samples]

    # Load the dataset
    splits = get_dataset(args.dataset)
    #
    # # Get the relevant dataset out
    dataset = splits["test"]["translation"][:args.n_samples]
    #
    targets = [d["en"] for d in dataset]
    # print(targets[:10])

    # Next we tokenize
    tokenized_samples = [
        tokenizer(samples, padding=False, )["input_ids"] for samples in list_of_samples]
    tokenized_targets = tokenizer(targets, padding=False)["input_ids"]

    # Lastly we gather the statistics
    results = {}
    for tokenized_sample, name in zip(tokenized_samples, args.refs_samples):
        n_grams = create_n_grams_count(tokenized_sample)
        length = [len(sample) for sample in tokenized_sample]
        results[name] = {"n_grams": n_grams, "lengths": length}

    n_grams = create_n_grams_count(tokenized_targets)
    length = [len(sample) for sample in tokenized_targets]
    results["targets"] = {"n_grams": n_grams, "lengths": length}

    print(results)
    print("Saving")
    ref = "./data/{}.pkl".format(args.name)
    save_pickle(results, ref)

    loaded_results = load_pickle("./data/{}.pkl".format(args.name))

if __name__ == '__main__':
    main()
