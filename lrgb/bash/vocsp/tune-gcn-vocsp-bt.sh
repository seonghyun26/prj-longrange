cd ../../
DATASET="vocsuperpixels_lg_bt"
model="GCN"
layer=("6" "8" "10" "12")
hdim=("252" "222" "198" "180")

# dropoutRate=0.7
for i in 1
do
    # seed 2 \
  python main.py \
    --cfg configs/tuned-backtrack/$DATASET-$model.yaml \
    --repeat 3 \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]}
    # gnn.dropout $dropoutRate \
    # gnn.self_loop False
  sleep 10
done