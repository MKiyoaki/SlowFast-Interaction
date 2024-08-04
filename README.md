# SlowFast-Interaction

### 1. Introduction

PySlowFast is basing on the framework of PySlowFast, 

### 2. Getting Start

#### 2.1. [Installation](./INSTALL.md)

#### 2.2. Data pre-process

1. Put the data into the corresponding folder. 
   - Default path is `./data`. 
   - Put the raw videos and labels into `./data/raw`
2. Change the initial path configurations at `/src/helper/configurations.py`
3. Call the [script](./src/helper/main.py) to perform video preprocess and dataset split. 

#### 2.3. Model Training

1. Get into the `/slowfast` folder at first.
```
cd slowfast
```
2. Using the following command to start a training process. 
```
python tools/run_net.py --cfg ../src/config/interaction_ua_2022_slowfast_config.yaml
```




