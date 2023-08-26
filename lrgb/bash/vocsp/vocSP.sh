cd ../../
DATASET="vocsuperpixels_lg"

python main.py --cfg configs/LG/vocsp/$DATASET-GCN.yaml \
  wandb.use True