cd ../

DATASET=$1
# peptides-func
# peptides-struct
# vocsuperpixels
model="GCN"
layer=("10" "15" "20")
hdim=("192" "162" "144")

for i in 0 1 2
do
  python main.py \
    --repeat 3 \
    --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.project lrgb-table \
    wandb.use True \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]}
  sleep 10
done