cd ../../
DATASET="peptides-func_lg"
model=$1
# for layer in 5 10 15 17 20
for layer in 25 30
do
  python main.py --cfg configs/LG_BB/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $layer \
    optim.max_epoch 200 \
    gnn.residual False 
  sleep 10
done