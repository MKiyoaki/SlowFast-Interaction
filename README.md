# SlowFast-Interaction

### 1. Installation

#### 1.1. Pre-requirements

1. Install Anaconda3
2. Using Anaconda to create and activate a new environment. 
  ```
  conda create -n slowfast python=3.8 -y
  conda activate slowfast
  ```

#### 1.2. Install PySlowFast

1. Please refer the [documentation](slowfast/INSTALL.md) to finish the preparation on pre requirements. 
   > - Python >= 3.8
   > - Numpy
   > - PyTorch >= 1.3
   > - [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
   > - [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation. You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
   > - simplejson: `pip install simplejson`
   > - GCC >= 4.9
   > - PyAV: `conda install av -c conda-forge`
   > - ffmpeg (4.0 is prefereed, will be installed along with PyAV)
   > - PyYaml: (will be installed along with fvcore)
   > - tqdm: (will be installed along with fvcore)
   > - iopath: `pip install -U iopath` or `conda install -c iopath iopath`
   > - psutil: `pip install psutil`
   > - OpenCV: `pip install opencv-python`
   > - torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
   > - tensorboard: `pip install tensorboard`
   > - moviepy: (optional, for visualizing video on tensorboard) `conda install -c conda-forge moviepy` or `pip install moviepy`
   > - PyTorchVideo: `pip install pytorchvideo`
   > - [Detectron2](https://github.com/facebookresearch/detectron2): `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`
   > - FairScale: `pip install 'git+https://github.com/facebookresearch/fairscale'`
2. Install PySlowFast
   ```
   cd slowfast
   pip install -e .
   ```

### 2. Getting started

#### 2.1. Data pre-process

1. Put the data into the corresponding folder. 
   - Default path is `./data`. 
   - Put the raw videos and labels into `./data/raw`
2. Change the initial path configurations at `/src/helper/configurations.py`
3. Call `/src/helper/main.py` to perform video preprocess and dataset split. 

#### 2.2. Model Training

1. Get into the `/slowfast` folder at first.
```
cd slowfast
```
2. Using the following command to start a training process. 
```
python tools/run_net.py --cfg ../src/config/interaction_config.yaml NUM_GPUS 1 TRAIN.BATCH_SIZE 8 SOLVER.BASE_LR 0.0125
```

#### 2.3. Edit a configuration file

### TBC

