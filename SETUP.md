# Setup Guide: Classify Unseen Objects Project

This project is to be setup on a Linux machine with a NVIDIA GPU.
The project implements a pipeline for object detection and discovery from pointcloud scenes and includes external projects for unsupervised object detection (UnScene3D) and feature extraction (Point-MAE). 

Setup Guide is a work in progress and not yet fully tested start-to-finish.

Currently working setups:
> Jakobs Desktop: /home/shared/ @ eu.loclx.io:5903
> CG-Lab Desktop: TODO @ cg.tuwien.ac.at 

---

## Prerequisites:

### Install Sources for Ubuntu 22.04 LTS:
- Define a project root directory (this is where where all code, drivers and dependencies will live, e.g. /home/user/Desktop): ```export $PROJECT_ROOT={path/to/project_root}```
- Download install sources:
- [CUDA Toolkit 11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive): ```wget -P $PROJECT_ROOT https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run```
- [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive): ```wget -P $PROJECT_ROOT https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run```
- [Miniconda 3](https://www.anaconda.com/docs/getting-started/miniconda/install): ```wget -P $PROJECT_ROOT https://repo.anaconda.com/miniconda/Miniconda3-py310_25.1.1-2-Linux-x86_64.sh```
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine): ```git clone https://github.com/NVIDIA/MinkowskiEngine.git $PROJECT_ROOT```

### Clone classify-unseen-objects repo:
- Clone repo: ```git clone TODO/classify-unseen-objects $PROJECT_ROOT```
- Initialize and update submodules: ```cd $PROJECT_ROOT/classify-unseen-objects; git submodule update --init --recursive```
- Checkout project structure: ```tree -L 3 $PROJECT_ROOT/classify-unseen-objects```

### Replace $PROJECT_ROOT Placeholders in .yml and .py files needing abosulte paths:
- Run the script to replace placeholders: ```bash $PROJECT_ROOT/classify-unseen-objects/pipeline_conf/project_root_placeholder_replace.sh $PROJECT_ROOT```


### Download ShapeNetCore:
- Run the download script: ```bash $PROJECT_ROOT/data/download-scripts/download_and_unzip.sh $PROJECT_ROOT```

### Download ScanNet Subset:
- Run the download script with the start and stop scene parameters: ```bash $PROJECT_ROOT/data/download-scripts/download_scannet_subset.sh $PROJECT_ROOT <start_scene> <stop_scene>```

### Setup Machine:
- Install g++-9 and gcc-9: ```sudo apt install gcc-9 g++-9```
- Install conda 25.1.1: ```bash Miniconda3-py310_25.1.1-2-Linux-x86_64.sh -b -p $PROJECT_ROOT```

### Setup Python
- Add conda 25.1.1 to path: ```export PATH=$PROJECT_ROOT/miniconda3/bin:$PATH```
- Check conda: ```which conda```
- Create conda environment: ```conda create -n classify-unseen-objects python=3.10 pip=25.0.1```
- Activate: ```conda activate classify-unseen-objects```
- Export conda root: ```export CONDA_ROOT={path/to/conda_root}```
- Add to PATH: ```export PATH=$CONDA_ROOT/envs/classify-unseen-objects/bin:$PATH```
- Check python and pip: ```which python; which pip```

### Remarks on chosen versions:
- Building MinkowskiEngine fails with Python >3.11 due to breaking changes for distutils (https://github.com/Julie-tang00/Common-envs-issues/blob/main/Cuda12-MinkowskiEngine, https://docs.python.org/3/whatsnew/3.10.html#distutils-deprecated).
- Nvidia driver 550.54.14 from CUDA Toolkit 12.4 is installed. Finally, CUDA Toolkit 11.6 is used but the included Nvidia driver 510.39.01 omitted, because of compatibility issues. This possible, because Nvidia drivers are backwards compatible (LINK).
- Miniconda3 is installed into $PROJECT_ROOT to prevent conflicts with other conda installations.

---

## Setup CUDA ([More infos](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)):
#### Install CUDA Toolkit 12.4 and Nvidia driver 550.54.14:
- Create installation directories for CUDA Toolkit 12.4: 
   ```export PROJECT_ROOT={path/to/project_root}```
   ```mkdir $PROJECT_ROOT/cuda-12.4```
   ```mkdir $PROJECT_ROOT/cuda-12.4/toolkit; mkdir $PROJECT_ROOT/cuda-12.4/defaultroot```
- Switch text-only terminal mode (tty): ```Ctrl + Alt + F3```
- Stop running XServer:
- Switch to multi-user target: ```sudo systemctl isolate multi-user.target```
- Check display manager in use: ```systemctl status display-manager```
- Stop display manager (e.g. lightdm, gdm, ly): ```sudo systemctl stop lightdm```
- Install CUDA Toolkit 12.4: and Nvidia driver 550.54.14: ```sudo sh cuda_12.4.0_550.54.14_linux.run --driver --toolkit --silent --toolkitpath=$PROJECT_ROOT/cuda-12.4/toolkit --defaultrootpath=$PROJECT_ROOT/cuda-12.4/defaultroot```
- Restart display manager (e.g. lightdm, gdm, ly): ```sudo systemctl start lightdm```
- Switch back to GUI session: ```Ctrl + Alt + F2```
#### Install CUDA Toolkit 11.6:
- Create installation directories for CUDA Toolkit 11.6:
   ```mkdir $PROJECT_ROOT/cuda-11.6```
   ```mkdir $PROJECT_ROOT/cuda-11.6/toolkit; mkdir $PROJECT_ROOT/cuda-11.6/defaultroot```
- Install CUDA Toolkit 11.6: ```sudo sh cuda_11.6.0_510.39.01_linux.run --toolkit --toolkitpath=$PROJECT_ROOT//cuda-11.6/toolkit --defaultrootpath=$PROJECT_ROOT/cuda-11.6/defaultroot```

## Setup PyTorch:
- Install PyTorch [compatible with CUDA Toolkit 11.6](https://pytorch.org/get-started/previous-versions/): ```pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116```

## Setup UnScene3D dependencies:
### Install python dependencies: (TODO move to final joint yaml)
- Install from yaml file: ```conda env update --file $PROJECT_ROOT/classify-unseen-objects/external/UnScene3D/conf/unscene3d_env.yml --prune```

### Setup MinkowskiEngine:
- Export environment variables:
   ```export CONDA_DEFAULT_ENV=classify-unseen-objects```
   ```export CONDA_PYTHON_EXE=$CONDA_ROOT/bin/python```
   ```export CONDA_PREFIX=$CONDA_ROOT/envs/classify-unseen-objects```
   ```export CONDA_PREFIX_1=$CONDA_ROOT/miniconda3```
   ```export LD_LIBRARY_PATH=$PROJECT_ROOT/cuda-11.6/toolkit/lib64```
   ```export CUDNN_LIB_DIR=$PROJECT_ROOT/cuda-11.6/toolkit/lib64```
   ```export CUDA_HOME=$PROJECT_ROOT/cuda-11.6/toolkit```
   ```export CXX=g++-9```
   ```export CC=gcc-9```
- Add to PATH: ```export PATH=/home/shared/cuda-11.6/toolkit/bin:/usr/bin:$PATH```
- Symlink g++ to g++-9: ```sudo ln -s /usr/bin/g++-9 /usr/bin/g++ ```
- Add to alternatives system: ```sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60; sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 60```
- Configure alternatives to use gcc-9: ```sudo update-alternatives --config gcc```
- Conifgure alternatives to use g++-9: ```sudo update-alternatives --config g++```
- Check versions: ```gcc --version; g++ --version```
- Install openblas-devel: ```conda install openblas-devel -c anaconda```
- Change to MinkowskiEngine git-repo directory: ```cd $PROJECT_ROOT/MinkowskiEngine```
- Build and install MinkowskiEngine: ```python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas```

### Setup Detectron2:
- Install Detectron2: ```python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'```

### Setup UnScene3D Utils:
- Change to C++ utils directory: ```cd $PROJECT_ROOT/classify-unseen-objects/external/UnScene3D/utils/cpp_utils```
- Install C++ utils: ```python setup.py install```
- Change to CUDA utils directory: ```cd $PROJECT_ROOT/classify-unseen-objects/external/UnScene3D/utils/cuda_utils```
- Install CUDA utils: ```python setup.py install```
     
### Setup PointNet2:
- Change to PointNet2 directory: ```cd $PROJECT_ROOT/classify-unseen-objects/external/UnScene3D/third_party/pointnet2```
- Install PointNet2: ```python setup.py install```

### Install other UnScene3D dependencies (TODO: Move to a final joint .yml later):
```pip install open3d```
```pip install albumentations```
```pip install hydra-core --upgrade```
```pip install torch-scatter```
```pip install numpy==1.26.4```

## Setup PointMAE dependencies (WIP, NOT FINISHED):

### Install Point-MAE dependencies:
- ```pip install -r $PROJECT_ROOT/classify-unseen-objects/external/PointMAE/requirements.txt```

### Setup PointNet++
- ```pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"```

### GPU kNN
- ```pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl```
- ```pip install emd```