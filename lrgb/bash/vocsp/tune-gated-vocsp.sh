cd ../../
DATASET="vocsuperpixels_lg"
model="GatedGCN"
layer=("6" "8" "10" "12" "14")
hdim=("120" "108" "96" "84" "72")

dropout=0.25
for i in 2 3 4
do
  python main.py \
    --repeat 3 \
    --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropout
  sleep 10
done