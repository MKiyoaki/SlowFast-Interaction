# SlowFast-Interaction

### 1. Introduction

SlowFast-Interaction is basing on the framework of PySlowFast, aims to explore the application of explainable AI paradigm (e.g. GradCAM) in detecting and classifying human's behavioural responses to different types of Interaction Ruptures. 

### 2. Getting Start

#### 2.1. [Installation](./INSTALL.md)

#### 2.2. Data pre-process

1. Put the data into the corresponding folder. 
   - Default path is `./data`. 
   - Put the raw videos and labels into `./data/raw`
2. Change the initial path configurations at the configuration files [`src/helper/configurations.py`](src/helper/configurations.py) and `src/configs`. 
3. Call [`src/tools/data_preprocess.py`](src/tools/data_preprocess.py) to perform video preprocess and dataset split. The scripts in the folder `src/tools` can be executed directly for corresponding purposes. 
   ```
   python src/tools/data_preprocess.py
   ```
4. (Optional) Call [`src/tools/dataset_statistics.py`](src/tools/dataset_statistics.py) to check the distributions of the datasets after partition.
  ```
  python src/tools/dataset_statistics.py
  ```

#### 2.3. Model Training

1. At the root of the project, using the following command to start a training process. You can check more options at [`src/configs`](src/configs). For instance, 
   ```
   python slowfast/tools/run_net.py --cfg src/config/ua_2022_slowfast.yaml
   ```
2. (Optional) Monitor the training process via tensorboard. The original path to the logs is the following: 
   ```
   tensorboard --logdir outputs/interaction/2022/slowfast/ua
   ```

#### 2.4 Advanced Training Modification (Optional)

1. Tweak on the datasets. Please check [`slowfast/slowfast/datasets/interaction.py`](slowfast/slowfast/datasets/interaction.py) and [`slowfast/slowfast/datasets/interaction_variant.py`](`slowfast/slowfast/datasets/interaction_variant.py`). 
2. Implementation about model training pipeline. Please check [`slowfast/tools/train_net.py`](slowfast/tools/train_net.py). 
3. Register more features in the configuration table. Please check [`slowfast/slowfast/config/defaults.py`](slowfast/slowfast/config/defaults.py). 
4. Implementation about model visualization. Please check [`slowfast/tools/visualization.py`](slowfast/tools/visualization.py).
5. More details about slowfast module can be found at their [README.md](slowfast/README.md) file. 

#### 2.5. Obtaining the Results

1. If you set visualization and Grad-CAM as true during the training process in the configuration table, then there will be some results in the tensorboard after training. 
2. If you want to get the videos from Grad-CAM as outputs, please set an output path in the configuration table. 
3. After obtaining the outputs, please check [`src/tools/output_process.py`](src/tools/output_process.py) for details. e.g., Combine the video into a sequence, K-Means clustering, etc. 

### 3. Contact Information

yifeishi.1224@gmail.com

