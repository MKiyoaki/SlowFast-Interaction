# SlowFast-Interaction

### 1. Installation

#### 1.1. Pre-requirements

- Create a new environment
  `conda create env -f requirement.yaml`
- Activate the new environment
  `conda activate slowfast`

#### 1.2. Install PySlowFast

- Please refer the documentation at `/slowfast/INSTALL.md`

### 2. Getting started

#### 2.1. Model Training

#### 2.2. Edit a configuration file

### TBC

```
cd slowfast
python tools/run_net.py --cfg ../src/interaction_config.yaml NUM_GPUS 1 TRAIN.BATCH_SIZE 8 SOLVER.BASE_LR 0.0125
```