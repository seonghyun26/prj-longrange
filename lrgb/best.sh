DATASET="peptides-funclg"
model=$1
for i in 10 15 17 20 25
do
  python main.py --cfg configs/LG/$DATASET-GatedGCN+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $i \
    optim.max_epoch 300 \
    gnn.residual False
  sleep 10
done