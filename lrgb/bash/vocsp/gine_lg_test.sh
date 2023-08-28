cd ../../
DATASET="vocsuperpixels_lg"
model="GINE"

for dropoutrate in 0.0 0.1 0.2
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/LG/vocsp/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 10 \
    gnn.dim_inner 148 \
    gnn.dropout $dropoutrate
  sleep 10
done