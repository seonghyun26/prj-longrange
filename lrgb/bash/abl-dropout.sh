cd ../
DATASET="peptides-funclg"
model=$1
for abl in 0.01 0.1 0.3
do
  python main.py --cfg configs/LG/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    optim.max_epoch 300 \
    gnn.layers_mp 15 \
    gnn.dropout $abl
  sleep 10
done