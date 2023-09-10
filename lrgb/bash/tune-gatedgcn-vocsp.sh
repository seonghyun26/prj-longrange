cd ../

# DATASET=$1
# peptides-func
# peptides-struct
# vocsuperpixels
model="GatedGCN"
for DATASET in "vocsuperpixels"
do
  python main.py \
    --repeat 3 \
    --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.project lrgb-table \
    wandb.use True \
    optim.max_epoch 400
  sleep 10
done