cd ../../
DATASET="vocsuperpixels_lg"
model="GatedGCN"
layer=("6" "8" "10" "12" "14")
hdim=("120" "108" "96" "84" "72")

i=2
for dropout in 0.4 0.6
do
    # --repeat 3 \
  python main.py \
    --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.lgvariant 21 \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropout
  sleep 10
done