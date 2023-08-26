cd ../../
DATASET="vocsuperpixels_lg"
layer=("5" "10" "15")
hdim=("138" "108" "88")
length=${#layer[@]}

for ((i=0;i<length;i++))
do
  python main.py --repeat 3 \
    --cfg configs/LG/vocsp/$DATASET-GatedGCN.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.lgvariant 22 \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    optim.max_epoch 400
  sleep 10
done