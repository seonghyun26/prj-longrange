cd ../

DATASET="peptides-struct_lg"
model=$1
for layer in 5 15 25 10 17 20 25
do
  # python main.py --cfg configs/LG/$DATASET-$model.yaml \
  python main.py --repeat 3 \
    --cfg configs/LG/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $layer \
    optim.max_epoch 250 \
    gnn.residual False
  sleep 10
done