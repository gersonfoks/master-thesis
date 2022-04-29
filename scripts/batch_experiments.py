# File to train a bunch of models after eachother.
# Idea is to start a bunch of experiments and see the day after what worked
import argparse

from training.functions import train_estimation_model

parser = argparse.ArgumentParser(
    description='Train a bunch of models defined in the list of training ')

parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                    help='Set to true if you want to do a very quick run to see if everything works')

parser.set_defaults(on_hpc=False)

args = parser.parse_args()
# List of configs of models to train. Contains tupples (train_func, config)
list_of_training = [
    (train_estimation_model, './configs/estimation/unigram_f1/basic-reference-models/reference-model-pool.yml'),
    (train_estimation_model, './configs/estimation/unigram_f1/basic-reference-models/reference-model-pool-overfit.yml'),
    (train_estimation_model, './configs/estimation/unigram_f1/basic-reference-models/reference-model-attention.yml'),
    (train_estimation_model, './configs/estimation/unigram_f1/basic-reference-models/reference-model-attention-overfit.yml'),
    (train_estimation_model, './configs/estimation/unigram_f1/basic-reference-models/reference-model-lstm.yml'),
    (train_estimation_model, './configs/estimation/unigram_f1/basic-reference-models/reference-model-lstm-overfit.yml'),



]



for (func, config) in list_of_training:
    func(config, args.smoke_test)



