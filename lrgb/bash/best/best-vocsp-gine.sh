cd ../../
DATASET="vocsuperpixels_lg"
model="GINE"

python main.py \
    --repeat 3 \
    --cfg configs/best/$DATASET-$model.yaml \
    wandb.project lrgb-table \
    wandb.use True \
    gnn.lgvariant 21 \
    gnn.layers_mp 12 \
    gnn.dim_inner 132