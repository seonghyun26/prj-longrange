cd ../../

DATASET="peptides-struct"
for model in "GCN" "GCNII" "GINE" "GatedGCN"
do
  for layer in 5 10 15 17 20 25
  do
    python main.py --repeat 3 \
      --cfg configs/$model/$DATASET-$model+LapPE.yaml \
      wandb.use True \
      wandb.project lrgb \
      gnn.layers_mp $layer \
      optim.max_epoch 250 \
      gnn.residual False
    sleep 10
  done
done