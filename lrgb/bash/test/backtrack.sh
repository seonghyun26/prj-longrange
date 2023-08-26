cd ../../
DATASET="peptides-func_lg_backtrack"
model=$1
for i in 5 10 15 25
do
  python main.py --cfg configs/LG_backtrack/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $i \
    optim.max_epoch 250 \
    gnn.residual False
  sleep 10
done