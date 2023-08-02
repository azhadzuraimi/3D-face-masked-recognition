"""Code was inspired by tbmoon's code from his 'facenet' repository
    https://github.com/tbmoon/facenet/blob/master/datasets/write_csv_for_making_dataset.ipynb

    The code was modified to run much faster since 'dataframe.append()' creates a new dataframe per each iteration
    which significantly slows performance.
"""

import os
import glob
import pandas as pd
import time
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Generating csv file for triplet loss!")
parser.add_argument('--dataroot', '-d', type=str, default = "test_file",
                    help="(REQUIRED) Absolute path to the dataset folder to generate a csv file containing the paths of the training images for triplet loss training."
                    )
parser.add_argument('--csv_name', type=str, default = "TestCustom.csv",
                    help="Required name of the csv file to be generated. (default: 'BosphorusCustom.csv')"
                    )
args = parser.parse_args()


def generate_csv_file(dataroot, csv_name):
    """Generates a csv file containing the image paths of the glint360k dataset for use in triplet selection in
    triplet loss training.
    Args:
        dataroot (str): absolute path to the training dataset.
        csv_name (str): name of the resulting csv file.
    """
    print("\nLoading image paths ...")
    files = glob.glob(dataroot + "/*/*")

    start_time = time.time()
    list_rows = []

    print("Number of files: {}".format(len(files)))
    print("\nGenerating csv file ...")

    progress_bar = enumerate(tqdm(files))

    for file_index, file in progress_bar:

        face_id = os.path.basename(file).split('.')[0]
        face_label = os.path.basename(os.path.dirname(file))

        # Better alternative than dataframe.append()
        row = {'id': face_id, 'name': face_label}
        list_rows.append(row)

    dataframe = pd.DataFrame(list_rows)
    dataframe = dataframe.sort_values(by=['name', 'id']).reset_index(drop=True)

    # Encode names as categorical classes
    dataframe['class'] = pd.factorize(dataframe['name'])[0]
    dataframe.to_csv(path_or_buf=csv_name, index=False)
    
    elapsed_time = time.time()-start_time
    print(f"the csv file is saved at {csv_name}")
    print("\nDone! Elapsed time: {:.2f} minutes.".format(elapsed_time/60))


if __name__ == '__main__':
    dataroot = os.path.join(os.path.dirname(os.path.abspath(__file__)),args.dataroot )
    csv_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),args.csv_name )
        
    if dataroot :
        print(f" the folder of data on this direcotry will be open on '{args.dataroot}'  ")
    if csv_name :
        print(f"the csv name is {args.csv_name}")

    
    generate_csv_file(dataroot=dataroot, csv_name=csv_name)
    