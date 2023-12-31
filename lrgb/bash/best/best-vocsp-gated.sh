cd ../../
DATASET="vocsuperpixels_lg"
model="GatedGCN"

python main.py \
    --repeat 3 \
    --cfg configs/best/$DATASET-$model.yaml \
    wandb.project lrgb-table \
    wandb.use True \
    gnn.lgvariant 21