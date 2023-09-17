cd ../../
DATASET="peptides-struct_lg_bb"
model="GCN"
layer=("5" "10" "15" "20" "25")
hdim=("132" "96" "78" "66" "60")

for i in 1 3
do
  python main.py \
    --repeat 3 \
    --cfg configs/tuned/$DATASET-$model-wolp.yaml \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]}
  sleep 10
done