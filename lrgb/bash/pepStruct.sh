cd ../
DATASET="peptides-struct"
model=$1
for layer in 5 10 15 17 20 25 30
do
  python main.py --cfg configs/GatedGCN/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $layer \
    optim.max_epoch 400 \
    gnn.residual False
  sleep 10
done

DATASET="peptides-struct_lg"
model=$1
for layer in 5 10 15 17 20 25 30
do
  python main.py --cfg configs/LG/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $layer \
    optim.max_epoch 250 \
    gnn.residual False
  sleep 10
done