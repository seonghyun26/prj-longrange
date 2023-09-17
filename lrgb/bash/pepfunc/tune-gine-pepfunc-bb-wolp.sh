cd ../../
DATASET="peptides-func_lg_bb"
model="GINE"
layer=("5" "10" "15" "20" "25")
hdim=("204" "150" "120" "96" "96")

for i in 1 3
do
  # ${hdim[i]} \
  python main.py \
    --repeat 3 \
    --cfg configs/tuned/$DATASET-$model-wolp.yaml \
    wandb.project lrgb-table \
    wandb.use True \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]}
  sleep 10
done