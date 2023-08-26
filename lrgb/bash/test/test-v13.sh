cd ../../

DATASET="peptides-struct_lg"

for model in "GatedGCN"
do
  for layer in 5 15 25
  do  
    python main.py --cfg configs/LG/$DATASET-$model+LapPE.yaml \
      wandb.use True \
      wandb.project lrgb \
      gnn.layers_mp $layer \
      optim.max_epoch 400 \
      gnn.residual False \
      gnn.lgvariant 13
    sleep 10
  done
done