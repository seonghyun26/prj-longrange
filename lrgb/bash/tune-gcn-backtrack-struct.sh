cd ../

DATASET="peptides-struct"
# peptides-func
# peptides-struct
# vocsuperpixels

layer=("10" "15" "20")
hdim=("192" "162" "144")
model="GCN"
for i in 0 1 2
do
    # --repeat 3 \
  python main.py \
    --cfg configs/tuned-backtrack/$DATASET-$model.yaml \
    wandb.project lrgb-table \
    wandb.use True \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    optim.max_epoch 350
  sleep 10
done