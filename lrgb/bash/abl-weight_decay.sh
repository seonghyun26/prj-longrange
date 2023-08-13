cd ../
DATASET="peptides-funclg"
model=$1
for i in 5 15 25
do
  python main.py --cfg configs/LG/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $i \
    gnn.residual False \
    optim.max_epoch 250 \
    optim.weight_decay 0.01 
  sleep 10
done