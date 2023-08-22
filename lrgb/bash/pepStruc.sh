cd ../

DATASET="peptides-struct_lg"

for layer in 5 10 15 17 20 25
do
  for model in "GCN" "GCNII" "GINE" "GatedGCN"
  do
  python main.py --repeat 3 \
    --cfg configs/LG/$DATASET-$model.yaml \
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