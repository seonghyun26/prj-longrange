cd ../
DATASET="peptides-funclg"
model=$1
for abl in 16 32 64 128
do
  python main.py --cfg configs/LG/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 15 \
    optim.max_epoch 300 \
    gnn.dim_inner $abl
  sleep 10
done