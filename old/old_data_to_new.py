'''
Temp Helper file
'''
import ast
import os

from tqdm import tqdm

from utils.PathManager import get_path_manager
import pandas as pd

def main():
    path_manager = get_path_manager()
    old_dataset_dir = 'predictive/tatoeba-de-en\data/raw\old'
    target_dir = 'predictive/tatoeba-de-en\data/raw/'
    old_dataset_dir = path_manager.get_abs_path(old_dataset_dir)
    target_dir = path_manager.get_abs_path(target_dir)
    files = os.listdir(old_dataset_dir)



    for file in tqdm(files):
        path = old_dataset_dir + '/'+file
        print("working on: ", file)
        df = pd.read_csv(path, sep="\t" )
        df["old_samples"] = df["samples"].map(lambda x: list(ast.literal_eval(x).items()))

        df["samples"] = df["old_samples"].map(lambda x: [a[0] for a in x] )

        df["count"] = df["old_samples"].map(lambda x: [a[1] for a in x] )

        # Then save the df

        df = df.drop(["old_samples"], axis=1)

        save_path = target_dir + '/' + file
        df.to_csv(save_path, sep="\t", index=False,)







if __name__ == '__main__':
    main()
