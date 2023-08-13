cd ../
DATASET="peptides-func"
model=$1
for i in 30
do
  python main.py --cfg configs/GatedGCN/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $i \
    optim.max_epoch 300
  sleep 10
done