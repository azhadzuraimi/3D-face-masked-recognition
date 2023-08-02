"""This module was imported from liorshk's 'facenet_pytorch' github repository:
        https://github.com/liorshk/facenet_pytorch/blob/master/LFWDataset.py

    It was modified to support point cloud files for loading by using the code
"""

"""MIT License

Copyright (c) 2017 liorshk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""


import torchvision.datasets as datasets
import os
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset


def normalize_pointcloud(pointcloud):
    # Normalizing sampled point cloud.
    norm_point_cloud = pointcloud - np.mean(pointcloud, axis=0)
    norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud, axis=1))
    return norm_point_cloud 

class TestBosphorus(Dataset):
    def __init__(self, dir, pairs_path):

        self.pairs_path = pairs_path

        # LFW dir contains 2 folders: faces and lists
        self.validation_points = self.get_Bosphorus_paths(dir)

    def read_lfw_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[:]:
                pair = line.strip().split()
                pairs.append(pair)

        return np.array(pairs, dtype=object)

    def get_Bosphorus_paths(self, Bosphorus_dir):
        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []
        for pair in pairs:
            if len(pair) == 3:
                path0 = self.add_extension(os.path.join(Bosphorus_dir, str(pair[0]), str(pair[1])))
                path1 = self.add_extension(os.path.join(Bosphorus_dir, str(pair[0]), str(pair[2])))
                issame = True
            elif len(pair) == 4:
                path0 = self.add_extension(os.path.join(Bosphorus_dir, str(pair[0]), str(pair[1])))
                path1 = self.add_extension(os.path.join(Bosphorus_dir, str(pair[2]), str(pair[3])))
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                path_list.append((path0, path1, issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list

    # Modified here
    def add_extension(self, path):
        if os.path.exists(path + '.ply'):
            return path + '.ply'
        else:
            raise RuntimeError('No file "%s" with extension ply.' % path)

    def __getitem__(self, index):
        """
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        """

        (path_1, path_2, issame) = self.validation_points[index]
        
        pcd = o3d.io.read_point_cloud(path_1)
        points_1=  np.asarray(pcd.points)
        pcd = o3d.io.read_point_cloud(path_2)
        points_2 =  np.asarray(pcd.points)
        
        points_1 = normalize_pointcloud(points_1)
        points_2 = normalize_pointcloud(points_2)
        points_1 = points_1.astype('float32')
        points_2 = points_2.astype('float32')
        return points_1, points_2, issame

    def __len__(self):
        return len(self.validation_points)


