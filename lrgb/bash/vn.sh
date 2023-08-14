cd ../
DATASET="peptides-func_lg_vn"
model=$1
for i in 10 15 17
do
  python main.py --cfg configs/LG_VN/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $i \
    optim.max_epoch 200 \
    gnn.residual False \
    train.batch_size 32
  sleep 10
done