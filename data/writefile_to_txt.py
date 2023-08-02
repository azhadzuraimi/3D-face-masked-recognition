import os
import glob
import pandas as pd
import time
import argparse
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import pymeshlab
import numpy as np
import open3d as o3d
import random 
 

parser = argparse.ArgumentParser(description="Generating test file for triplet loss")
parser.add_argument('--dataroot', '-d', type=str, default = None,
                    help="(REQUIRED) Absolute path to the dataset folder to generate a csv file containing the paths of the training images for triplet loss training."
                    )
parser.add_argument('--output_folder', type=str, default = "test_txt",
                    help="Required name of the csv file to be generated. (default: 'Bosphorus.csv')"
                    )
parser.add_argument('--samples', type=int, default = 2,
                    help="number of samples"
                    )
args = parser.parse_args()
dataroot = args.dataroot
output_folder = args.output_folder
samples = args.samples


 
def random_pairs(number_list): 
    return [number_list[i] for i in random.sample(range(len(number_list)), 2)] 
 

def generate_test_text_file(dataroot, output_folder, samples, filename = "Bosphorus_pair.txt"):
    """Generates a text file containing the image paths of the glint360k dataset for use in triplet selection in
    triplet loss training.

    Args:
        dataroot (str): absolute path to the training dataset.
        csv_name (str): name of the resulting csv file.
    """
    print("\nLoading image paths ...")
    files = glob.glob(dataroot + "/*/*")
   
    outputfolder = os.path.join(os.path.dirname(dataroot), output_folder)
    if not os.path.isdir(outputfolder):
        os.makedirs(outputfolder)
    outputfile = os.path.join(outputfolder,filename)
                
    SAMPLES = samples #2    
    start_time = time.time()
    face_classes = dict()
    face_ids = []
    face_labels = []
    face_max_idx = []
    face_last_idx = []
    last_idx = 0
    txt_all = []
    unique_class = []
    
    print("Number of files: {}".format(len(files)))
    print("\nGenerating txt file ...")

    progress_bar = enumerate(tqdm(files))
    for file_index, file in progress_bar:

        face_id = os.path.basename(file).split('.')[0]
        face_label = os.path.basename(os.path.dirname(file))
        face_ids.append(face_id)
        face_labels.append(face_label)

    for idx, label in enumerate(face_labels):
        if label not in face_classes:
            face_classes[label] = []
        face_classes[label].append(face_ids[idx])
        
    for i in face_classes:
        face_max_idx.append(len(face_classes[i]))
        last_idx += len(face_classes[i])
        face_last_idx.append(last_idx)
        
    for key in face_classes:
        unique_class.append(key)
        
    for i in range (len(face_classes)):
            
        numbers = list(np.arange(face_max_idx[i]))
        pairs = [random_pairs(numbers) for l in range(8)] 
        for j in range (4):
            txt = []
            txt.append(unique_class[i])
            idx = face_last_idx[i] - pairs[j][0]
            txt.append(face_ids[idx-1])
            idx = face_last_idx[i] - pairs[j][1]
            txt.append(face_ids[idx-1])
            txt_all.append(txt)
            
        random_class = np.random.choice(len(face_classes), 4, replace=False)
        while i in random_class:
            random_class = np.random.choice(len(face_classes), 4, replace=False)    
        for k in range (4):
            numbers = list(np.arange(face_max_idx[random_class[k]]))
            pairs2 = [random_pairs(numbers) for l in range(1)]        
            txt = []
            txt.append(unique_class[i])
            idx = face_last_idx[i] - pairs[k+4][0]
            txt.append(face_ids[idx-1])
            txt.append(unique_class[random_class[k]])
            idx = face_last_idx[random_class[k]] - pairs2[0][1]
            txt.append(face_ids[idx-1])
            txt_all.append(txt)
        
    print("hello")
    with open(outputfile, 'w') as f:
        for value in txt_all:
            f.writelines ('\t'.join(value))
            f.write("\n")
               
            
            
            
            
        # for i in range(0, SAMPLES):
        
        #     file_path = os.path.join(dataroot, face_label, face_id + ".ply")
        #     ms = pymeshlab.MeshSet()
        #     ms.load_new_mesh(file_path)
        #     ms.generate_sampling_poisson_disk(samplenum = POINTS, exactnumflag = True)
        #     ms.set_current_mesh(1)
        #     m = ms.current_mesh()
        #     v_matrix = m.vertex_matrix()
        #     ms.set_current_mesh(0)
        #     ms.generate_sampling_poisson_disk(samplenum = POINTS, exactnumflag = True)
        #     ms.set_current_mesh(2)
        #     m = ms.current_mesh()
        #     v_matrix_2 = m.vertex_matrix()
        #     fix_v_matrix = np.concatenate((v_matrix, v_matrix_2),axis=0)
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(np.asarray(fix_v_matrix[:2048]))
        #     file_output =  os.path.join(outputfile, face_label)
        #     if not os.path.isdir(file_output):
        #         os.makedirs(file_output)
        #     file_output =  os.path.join(outputfile, face_label, face_id + "_"+str(i)+".ply")
        #     o3d.io.write_point_cloud(file_output, pcd)
            
        
    
    
if __name__ == '__main__':
    if output_folder :
        print(f"the csv name is {output_folder}")
    baseroot = os.path.dirname(os.path.abspath(__file__))
    dataroot = os.path.join(baseroot,"test_file")
    print(os.path.exists(dataroot))
     
    generate_test_text_file(dataroot=dataroot, output_folder=output_folder, samples=samples)
    
# with open('logs/{}_log_triplet_batch{}.txt'.format(model_architecture,batch_size), 'a') as f:
#             val_list = [
#                 np.mean(running_loss),
#                 epoch,
#                 num_valid_training_triplets
#             ]
#             log = '\t'.join(str(value) for value in val_list)
#             f.writelines(log + '\n')