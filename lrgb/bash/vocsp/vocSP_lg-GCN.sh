cd ../../
DATASET="vocsuperpixels_lg"
layer=("5" "10" "15")
hdim=("220" "200" "168")

for ((i=0;i<length;i++))
do
  python main.py --cfg configs/LG/vocsp/$DATASET-GCN.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.lgvariant 20 \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]}
  sleep 10
done