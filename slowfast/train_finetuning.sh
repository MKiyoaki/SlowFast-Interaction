python tools/run_net.py --cfg ../src/config/interaction_multi_2022_slowfast_config.yaml
python tools/run_net.py --cfg ../src/config/interaction_multi_2022_slowfast_config.yaml \
  DATA.PATH_TO_DATA_DIR "/media/kiyoaki/Ventoy/projects/SlowFast-Interaction/data/interaction/clip/2022/w1" \
  SOLVER.BASE_LR 0.006 \
  SOLVER.MAX_EPOCH 10 \
  WARMUP_EPOCHS 0.0 \
  TEST.ENABLE True \
  TENSORBOARD.MODEL_VIS.ENABLE True
