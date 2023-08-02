import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(BASE_DIR,"train_file") 


def read_lfw_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)

    return np.array(pairs, dtype=object)
    
def add_extension(path):
    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)
        
def get_lfw_paths(lfw_dir, pairs_path ):
    pairs = read_lfw_pairs(pairs_path)

    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list.append((path0, path1, issame))
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list
    
if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pairs_path = os.path.join(BASE_DIR,"LFW_pairs.txt") 
    lfw_dir = os.path.join(BASE_DIR,"lfw_224")
    
    validation_images = get_lfw_paths(lfw_dir, pairs_path)
    
    for i in range (10):
        (path_1, path_2, issame) = validation_images[i]
        print(path_1, path_2, issame)
    
        
    
