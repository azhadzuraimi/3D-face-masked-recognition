# 3D-face-masked-recognition
Official PyTorch implementation of "Learning similarity and dissimilarity in 3D face masked with PointMLP, PointNet++ and PointNet triplet network"

__Operating System__: Ubuntu 18.04 (you may face issues importing the packages from the requirements.yml file if your OS differs).

## Install

```bash
# step 1. clone this repo
git clone https://github.com/azhadzuraimi/3D-face-masked-recognition.git
cd 3D-face-masked-recognition

# step 2. create a conda virtual environment and activate it
conda create --name <environment_name> --file requirements.txt
conda activate <environment_name>
```

```bash
# Optional solution for step 2: install libs step by step
conda create -n <environment_name> python=3.7 -y
conda activate <environment_name>
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=10.2 -c pytorch -y
pip install cycler einops h5py pyyaml==5.4.1 scikit-learn==0.24.2 scipy tqdm matplotlib==3.4.2
pip install pointnet2_ops_lib/.
```

## Usage

### Classification ModelNet40


## Acknowledgment

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.
[facenet-pytorch-glint360k](https://github.com/tamerthamoqa/facenet-pytorch-glint360k/tree/master)
[pointMLP-pytorch](https://github.com/ma-xu/pointMLP-pytorch)
[Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
[Pointnet_pytorch](https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-pytorch)

### Hardware Specifications
* NVIDIA GTX 1660ti Graphics Card (6 gigabytes Video RAM).
* i5-10400 Intel CPU.
* 32 Gigabytes DDR4 RAM at 3600 MHz.
## LICENSE
PointMLP is under the Apache-2.0 license. 