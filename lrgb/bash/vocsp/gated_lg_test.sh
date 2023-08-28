cd ../../
DATASET="vocsuperpixels_lg"
model="GatedGCN"

for dropoutrate in 0.2 0.1
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/LG/vocsp/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 10 \
    gnn.dim_inner 94 \
    gnn.residual False \
    gnn.batchnorm True \
    gnn.dropout $dropoutrate \
    optim.max_epoch 150
  sleep 10
done