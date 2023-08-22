cd ../
DATASET="peptides-func"
model=$1

for layer in "GCN" "GCNII" "GINE"
do
  # python main.py --repeat 3 \
  #   --cfg configs/$layer/$DATASET-$layer.yaml \
  #   wandb.use True \
  #   wandb.project lrgb \
  #   train.batch_size 64 \
  #   gnn.layers_mp 15 \
  #   gnn.residual False
  
  # sleep 10

  python main.py --repeat 3 \
    --cfg configs/$layer/$DATASET-$layer+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    optim.max_epoch 400 \
    gnn.layers_mp 15 \
    gnn.residual False
  sleep 10
done