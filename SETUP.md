# Setup Guide: Classify Unseen Objects Project

This project is to be setup on a Linux machine with a NVIDIA GPU.
The project implements a pipeline for object detection and discovery from pointcloud scenes and includes external projects for unsupervised object detection (UnScene3D) and feature extraction (Point-MAE). 

Setup Guide is a work in progress and not yet fully tested start-to-finish.

Currently working setups:
>- Jakob-Linux Desktop: $PROJECT_ROOT=/home/shared/ @ eu.loclx.io:5903
<!--- >- CG-Lab Desktop: TODO @ cg.tuwien.ac.at  -->

### Remarks on setup guide and chosen dependency versions:
- This setup guide includes our *recommended* installation of specific local cuda toolkit/nvidia drivers and miniconda installation. Diverging from this (e.g. using your existing global conda installation and cuda toolkit/nvidia driver) requires adjusting setup steps.
- Building MinkowskiEngine [fails with Python >3.11](https://github.com/Julie-tang00/Common-envs-issues/blob/main/Cuda12-MinkowskiEngine) due to [breaking changes for distutils](https://docs.python.org/3/whatsnew/3.10.html#distutils-deprecated). Therefor we use Python 3.10.
- CUDA Toolkit 11.6 is installed but Nvidia driver 510.39.01 is omitted, because of compatibility issues. Instead Nvidia driver 550.54.14 from CUDA Toolkit 12.4 is installed. This possible, because Nvidia drivers are backwards compatible ([CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/#why-cuda-compatibility), [Versioning and Compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#versioning-and-compatibility)).
- A local Miniconda3 is installed to prevent conflicts with other conda installations.
- For Point-MAE, building and installing ```emd``` and ```chamfer_dist``` as recommended in the original repo is omitted to avoid version conflicts for now. This works, as Point-MAE is currently not used for training but only used for inference.
---

## 1. Repository Setup:

#### Export environment variable for $PROJECT_ROOT as *absolute* path of parent directory of 'classify-unseen-objects' repository location
- All dependencies will be installed into $PROJECT_ROOT location: 
    - ```export PROJECT_ROOT={absolute/path/to/project_root}```
    - ```cd $PROJECT_ROOT``` 

#### Initialize classify-unseen-objects submodules:
- Initialize and update submodules: 
    - ```cd $PROJECT_ROOT/classify-unseen-objects```
    - ```git submodule update --init --recursive```
- Checkout project structure: 
    - ```tree -L 3 $PROJECT_ROOT/classify-unseen-objects```


#### Update $PROJECT_ROOT placeholders:
- Multiple files contain placeholders for $PROJECT_ROOT. They can be automatically replaced by running: 
    - ```bash $PROJECT_ROOT/classify-unseen-objects/pipeline_conf/project_root_placeholder_replace.sh $PROJECT_ROOT```

#### Download datasets:

- **ShapeNetCore**: [ShapeNetCore](https://www.kaggle.com/datasets/jeremy26/shapenet-core) can be automatically downloaded into ```$PROJECT_ROOT/classify-unseen-objects/data/shapenetcore``` by running: 
    - ```bash $PROJECT_ROOT/classify-unseen-objects/data/download-scripts/download_shapenetcore.sh $PROJECT_ROOT```
<br>

- **ScanNet**: Subsets of [ScanNet](http://www.scan-net.org/) scenes can be automatically downloaded into ```$PROJECT_ROOT/classify-unseen-objects/data/scannet/scannet_scenes/``` by running: 
    - ```bash $PROJECT_ROOT/classify-unseen-objects/data/download-scripts/download_scannet.sh $PROJECT_ROOT <start_scene> <stop_scene>```

---

## 2. System Setup:

#### Prerequisites:
- Install g++-9 and gcc-9: 
    - ```sudo apt install gcc-9 g++-9```
- Install [Miniconda3](https://www.anaconda.com/docs/getting-started/miniconda/install) 25.1.1 and add to PATH: 
    - ```wget -P $PROJECT_ROOT https://repo.anaconda.com/miniconda/Miniconda3-py310_25.1.1-2-Linux-x86_64.sh```
    - ```bash Miniconda3-py310_25.1.1-2-Linux-x86_64.sh -b -p $PROJECT_ROOT```
    - ```export PATH=$PROJECT_ROOT/miniconda3/bin:$PATH```
    - ```which conda```
- Create conda environment with Python 3.10 and pip 25.0.1 and activate: 
    - ```conda create -n classify-unseen-objects python=3.10 pip=25.0.1```
    - ```conda activate classify-unseen-objects```
- Export conda root and add environment bin to PATH
    - ```export CONDA_ROOT={path/to/conda_root}```
    - ```export PATH=$CONDA_ROOT/envs/classify-unseen-objects/bin:$PATH```
    - ```which python; which pip```

---

## 3. CUDA Setup:
More information in official [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

#### Check Nvidia driver installation:
- ```nvidia-smi```

#### If ```nvidia-smi``` does not find Nvidia driver or driver is older than 510.39.01, install Nvidia driver 550.54.14 from CUDA Toolkit 12.4:
>- TODO: Add instructions to uninstall old Nvidia driver, depending on if it is runfile or package manager installation.
- Download [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive) runfile:
    - ```wget -P $PROJECT_ROOT https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run```
- Switch text-only terminal mode (tty), switch to multi-user target, check which display manager is running and stop it: 
    - ```Ctrl + Alt + F3```
    - ```sudo systemctl isolate multi-user.target```
    - ```systemctl status display-manager```
    - ```sudo systemctl stop {display_manager}```
- Install Nvidia driver 550.54.14 from CUDA Toolkit 12.4 runfile : 
    - ```sudo sh cuda_12.4.0_550.54.14_linux.run --driver --silent``` 
- Restart display manager and switch back to GUI: 
    - ```sudo systemctl start {display_manager}```
    - ```Ctrl + Alt + F2```

<!--
- Create installation directories for CUDA Toolkit 12.4: 
   ```export PROJECT_ROOT={path/to/project_root}```
   ```mkdir $PROJECT_ROOT/cuda-12.4```
   ```mkdir $PROJECT_ROOT/cuda-12.4/toolkit; mkdir $PROJECT_ROOT/cuda-12.4/defaultroot```
-->
<!--
toolkitpath=$PROJECT_ROOT/cuda-12.4/toolkit --defaultroot=$PROJECT_ROOT/cuda-12.4/defaultroot
-->

#### Install CUDA Toolkit 11.6:
- Download [CUDA Toolkit 11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive): runfile:
    - ```wget -P $PROJECT_ROOT https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run```
- Create installation directories for CUDA Toolkit 11.6:
   - ```mkdir $PROJECT_ROOT/cuda-11.6```
   - ```mkdir $PROJECT_ROOT/cuda-11.6/toolkit```
   - ```mkdir $PROJECT_ROOT/cuda-11.6/defaultroot```
- Install CUDA Toolkit 11.6: 
    - ```sudo sh cuda_11.6.0_510.39.01_linux.run --toolkit --toolkitpath=$PROJECT_ROOT/cuda-11.6/toolkit --defaultroot=$PROJECT_ROOT/cuda-11.6/defaultroot```
- Add CUDA Toolkit 11.6 bin to PATH:
    - ```export PATH=/home/shared/cuda-11.6/toolkit/bin:$PATH```
---

## 4. Python Dependencies Setup:

#### 4.1 Classify Unseen Objects dependencies:
##### Install required python dependencies:
>TODO: This should be a unified requirements.txt/env.yml later
- Other dependencies:    
    - ```pip install pyqt5;```
    - ```pip install vtk;```
    - ```pip install umap-learn```

#### 4.2 UnScene3D dependencies:

##### Install required python dependencies:
>TODO: This should be a unified requirements.txt/env.yml later
- From UnScene3D .yml file: 
    - ```conda env update --file $PROJECT_ROOT/classify-unseen-objects/external/UnScene3D/conf/unscene3d_env.yml --prune```
- Other dependencies:    
    - ```pip install open3d;```
    - ```pip install albumentations;```
    - ```pip install hydra-core --upgrade;```
    - ```pip install torch-scatter;```
    - ```pip install numpy==1.25.1;```
    - ```pip install albumentations;```
    - ```pip install wandb;```
    - ```pip install torchmetrics==1.1.0;```
    - ```pip install easydict```
##### Install PyTorch:
- Install PyTorch 1.13.1 [compatible with CUDA Toolkit 11.6](https://pytorch.org/get-started/previous-versions/): 
    - ```pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116```

##### Build and install MinkowskiEngine:
- Clone [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) repo: 
    - ```git clone https://github.com/NVIDIA/MinkowskiEngine.git $PROJECT_ROOT```
- Export relevant environment variables:
    - ```export CONDA_DEFAULT_ENV=classify-unseen-objects;``` 
    - ```export CONDA_PYTHON_EXE=$CONDA_ROOT/bin/python;```  
    - ```export CONDA_PREFIX=$CONDA_ROOT/envs/classify-unseen-objects;``` 
    - ```export CONDA_PREFIX_1=$CONDA_ROOT/miniconda3;``` 
    - ```export LD_LIBRARY_PATH=$PROJECT_ROOT/cuda-11.6/toolkit/lib64;``` 
    - ```export CUDNN_LIB_DIR=$PROJECT_ROOT/cuda-11.6/toolkit/lib64;``` 
    - ```export CUDA_HOME=$PROJECT_ROOT/cuda-11.6/toolkit;``` 
    - ```export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5";```
    - ```export CXX=g++-9;``` 
    - ```export CC=gcc-9```
- Symlink g++ to g++-9: 
    - ```sudo ln -s /usr/bin/g++-9 /usr/bin/g++```
- Add gcc-9 and g++-9 to alternatives system: 
    -  ```sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60;```
    -  ```sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 60```
- Configure alternatives to use gcc-9:
    - ```sudo update-alternatives --config gcc```
- Conifgure alternatives to use g++-9: 
    - ```sudo update-alternatives --config g++```
- Sanity check versions: 
    - ```gcc --version;```
    - ```g++ --version```
- Install openblas-devel: 
    - ```conda install openblas-devel -c anaconda```
- Build and install to MinkowskiEngine: 
    - ```cd $PROJECT_ROOT/MinkowskiEngine;```
    - ```python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas```
- Reconfigure alternatives to use previously used gcc version:
    - ```sudo update-alternatives --config gcc```
- Reconifgure alternatives to use previously used g++ version: 
    - ```sudo update-alternatives --config g++```

##### Build and install Detectron2:
- ```python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'```

##### Build and install UnScene3D Utils:
- C++ utils: 
    - ```cd $PROJECT_ROOT/classify-unseen-objects/external/UnScene3D/utils/cpp_utils;```
    - ```python setup.py install```
- CUDA utils: 
    - ```cd $PROJECT_ROOT/classify-unseen-objects/external/UnScene3D/utils/cuda_utils;```
    - ```python setup.py install```
     
##### Build and install PointNet2:
- ```cd $PROJECT_ROOT/classify-unseen-objects/external/UnScene3D/third_party/pointnet2;```
- ```python setup.py install```

#### 4.3 PointMAE dependencies:

##### Install required python dependencies:
>TODO: This should be a unified requirements.txt/env.yml later
- From Point-MAE requirements.txt:
    - ```pip install -r $PROJECT_ROOT/classify-unseen-objects/external/PointMAE/requirements.txt```
- Other dependencies:    
    - ```pip install emd;```
    - ```pip install timm;```
    - ```pip install loguru```
##### Install pointnet2_ops
- ```pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"```

##### Install GPU kNN
- ```pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl```
