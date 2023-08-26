cd ../../
DATASET="vocsuperpixels_lg"
model="GCN"
for layer in 15
do
  python main.py --cfg configs/LG/vocsp/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 15 \
    gnn.dim_inner 168
  sleep 10
done