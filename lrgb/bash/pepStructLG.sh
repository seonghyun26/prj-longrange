cd ../

DATASET="peptides-struct_lg"
model=$1
for i in 5 15 25 10 17 20
do
  # python main.py --cfg configs/LG/$DATASET-$model.yaml \
  python main.py --cfg configs/LG/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $i \
    optim.max_epoch 250 \
    gnn.residual False
  sleep 10
done