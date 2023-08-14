cd ../
DATASET="peptides-funclg"
model=$1
for i in 20
do
  python main.py --repeat 3 \
    --cfg configs/LG/$DATASET-GatedGCN+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $i \
    optim.max_epoch 300 \
    gnn.residual False \
    seed 1
  sleep 10
done