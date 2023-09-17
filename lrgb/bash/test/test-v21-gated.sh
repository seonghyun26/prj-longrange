cd ../../

model="GatedGCN"
DATASET="vocsuperpixels_lg"

python main.py --cfg configs/best/$DATASET-$model.yaml \
  wandb.use True \
  wandb.project lrgb \
  gnn.lgvariant 21