cd ../../
DATASET="peptides-funclg"
model=$1
for wd in 0.1
do
  for layer in 15 17 20
  do
    python main.py --cfg configs/LG/$DATASET-$model.yaml \
      wandb.use True \
      wandb.project lrgb \
      gnn.layers_mp $layer \
      gnn.residual False \
      optim.max_epoch 200 \
      optim.weight_decay $wd
    sleep 10
  done
done