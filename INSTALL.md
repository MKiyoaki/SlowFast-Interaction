# SlowFast-Interaction

### 1. Pre-requirements

1. Install Anaconda3
2. Using Anaconda to create and activate a new environment. 
  ```
  conda create -n slowfast python=3.8 -y
  conda activate slowfast
  ```

### 2. Install PySlowFast

1. Please refer the [documentation](slowfast/INSTALL.md) to finish the preparation on pre requirements. 
   > - Python >= 3.8
   > - Numpy
   > - PyTorch >= 1.3
   > - [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
   > - [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation. You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
   > - simplejson: `pip install simplejson`
   > - GCC >= 4.9
   > - ffmpeg: (4.0 is prefereed, will be installed along with PyAV) `conda install ffmpeg=4.2 -y`
   > - PyYaml: (will be installed along with fvcore)
   > - tqdm: (will be installed along with fvcore)
   > - iopath: `pip install -U iopath` or `conda install -c iopath iopath -y`
   > - psutil: `pip install psutil`
   > - OpenCV: `pip install opencv-python`
   > - tensorboard: `pip install tensorboard`
   > - moviepy: (optional, for visualizing video on tensorboard) `conda install -c conda-forge moviepy` or `pip install moviepy`
   > - PyTorchVideo: `pip install "git+https://github.com/facebookresearch/pytorchvideo.git"`
   > - [Detectron2](https://github.com/facebookresearch/detectron2): `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`
   > - FairScale: `pip install 'git+https://github.com/facebookresearch/fairscale'`
2. Install PySlowFast
   ```
   cd slowfast
   pip install -e .
   ```

3. Build PySlowFast
   ```
   python setup.py build develop
   ```