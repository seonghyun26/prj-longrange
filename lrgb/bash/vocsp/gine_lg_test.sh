cd ../../
DATASET="vocsuperpixels_lg"
model="GINE"
for layer in 15
do
  python main.py --repeat 3 \
    --cfg configs/LG/vocsp/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 5 \
    gnn.dim_inner 204
  sleep 10
done