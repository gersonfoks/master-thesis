import argparse
import ast
from tqdm.contrib import tzip
import pandas as pd
from datasets import tqdm


def text_to_file(text, file):
    with open(file, "w", encoding="utf8") as f:
        f.write(text)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Test an NMT model')
    parser.add_argument('--ref-dataset', type=str,
                        default='./data/validation_predictive_ancestral_100.csv',
                        help='The references to use')
    parser.add_argument('--hypothesis-dataset', type=str,
                        default='./data/validation_predictive_ancestral_100.csv',
                        help='The hypothesis to use')

    parser.add_argument("--save-output-base", type=str, default="./data/validation-helsinki-tatoeba")


    args = parser.parse_args()

    ref_save_file = args.save_output_base + "-ref.en"
    hyp_save_file = args.save_output_base + "-hyp.en"
    source_save_file = args.save_output_base + "-source.de"


    # save_file = args.dataset[:-4] + '_bayes_risk.csv'

    # Load the data
    ref_data = pd.read_csv(args.ref_dataset, sep="\t")
    hyp_data = pd.read_csv(args.hypothesis_dataset, sep="\t")
    # print(save_file)

    hyp_data["samples"] = hyp_data["samples"].map(lambda x: [*(ast.literal_eval(x)).keys()])
    ref_data["samples"] = ref_data["samples"].map(lambda x: [*(ast.literal_eval(x)).keys()])

#    print(ref_data["samples"][0])

    ref_data = ref_data.set_index("source")["samples"].apply(pd.Series).stack().reset_index(level=-1, drop=True).reset_index()
    hyp_data = hyp_data.set_index("source")["samples"].apply(pd.Series).stack().reset_index(level=-1, drop=True).reset_index()

    print("ref_data")
    print(ref_data)
    print("hyp_data")
    print(hyp_data)
    joined_ref_data = ref_data.merge(hyp_data, on="source")



    hyp_text = ""
    ref_text = ""
    source_text = ""


    #sources = joined_ref_data["source"]
    print(joined_ref_data)
    #sources.to_csv("./test.txt", header=None, index=None, sep=" ", mode="a")
    # for source, reference, hypothesis in tzip(joined_ref_data["source"], joined_ref_data["0_x"], joined_ref_data["0_y"]):
    #     source_text += source + "\n"
    #     hyp_text += hypothesis + "\n"
    #     ref_text += reference + "\n"
    #
    # text_to_file(hyp_text, hyp_save_file)
    # text_to_file(ref_text, ref_save_file)
    # text_to_file(source_text, source_save_file)
    #hyp_data["samples"] = hyp_data["samples"].map(lambda x: ast.literal_eval(x))

    # print(ref_data["samples"][0].keys())





    # for source, references, hypothesis in tzip(ref_data["source"], ref_data["samples"], hyp_data["samples"]):
    #     already_in = set()
    #
    #     hyps = [*hypothesis.keys()]
    #
    #
    #     for i_1 in range(len(hyps)):
    #         for i_2 in range(i_1, len(hyps)):
    #
    #             hyp = hyps[i_1]
    #             ref = hyps[i_2]
    #
    #             source_text += source + "\n"
    #             hyp_text += hyp + "\n"
    #             ref_text += ref + "\n"
    #
    #
    #
    #
    # text_to_file(hyp_text, hyp_save_file)
    # text_to_file(ref_text, ref_save_file)
    text_to_file(source_text, source_save_file)

if __name__ == '__main__':
    main()
