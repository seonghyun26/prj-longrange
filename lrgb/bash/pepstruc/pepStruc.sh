cd ../../

DATASET="peptides-struct_lg"

for layer in 15
do
  for model in "GCN" "GINE" "GCNII"
  do
  python main.py --cfg configs/LG/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $layer \
    gnn.residual False \
    gnn.linegraph True \
    gnn.lgvariant 12 \
    optim.max_epoch 400
  sleep 10
  done
done

for layer in 15
do
  for model in "GINE"
  do
  python main.py --cfg configs/LG/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $layer \
    gnn.residual False \
    gnn.linegraph True \
    gnn.lgvariant 13 \
    optim.max_epoch 400
  sleep 10
  done
done