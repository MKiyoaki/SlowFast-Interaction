# SlowFast-Interaction

### 1. Installation

#### 1.1. Pre-requirements

1. Install Anaconda3
2. Using Anaconda to install and activate the environment. 
  ```
  conda create env -f requirement.yaml \
  conda activate slowfast
  ```

#### 1.2. Install PySlowFast

1. Please refer the documentation at `/slowfast/INSTALL.md`

### 2. Getting started

#### 2.1. Data pre-process

1. Put the data into the corresponding folder. 
2. Change the initial path configurations at `/src/helper/configurations.py`
3. Call `/src/helper/main.py` to perform video preprocess and dataset split. 

#### 2.2. Model Training

1. Get into the `/slowfast` folder at first.
```
cd slowfast
```
2. Using the following command to start a training process. 
```
python tools/run_net.py --cfg ../src/interaction_config.yaml NUM_GPUS 1 TRAIN.BATCH_SIZE 8 SOLVER.BASE_LR 0.0125
```

#### 2.3. Edit a configuration file

### TBC

