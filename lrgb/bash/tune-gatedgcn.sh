cd ../

DATASET="peptides-func"
# peptides-func
# peptides-struct
# vocsuperpixels

layer=("15" "20" "25")
hdim=("78" "66" "60")
model="GatedGCN"
for i in 0 1 2
do
  python main.py \
    --repeat 3 \
    --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.project lrgb-table \
    wandb.use True \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    optim.max_epoch 350
  sleep 10
done