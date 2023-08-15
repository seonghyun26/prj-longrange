cd ../
DATASET="peptides-funclg"
model=$1
for gnn in "GCN" "GIN" "GCNII"
do
  python main.py --repeat 3 \
    --cfg configs/LG/$DATASET-$gnn+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    optim.max_epoch 200 \
    gnn.layers_mp 15 \
    gnn.residual False
  sleep 10
done