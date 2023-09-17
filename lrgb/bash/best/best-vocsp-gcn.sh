cd ../../

DATASET="vocsuperpixels_lg"
model="GINE"

python main.py \
    --repeat 3 \
    --cfg configs/best/$DATASET-$model.yaml \
    wandb.project lrgb-table \
    wandb.use True