cd ../
DATASET="peptides-func_lg_bb"
model="GINE"
layer=("5" "10" "15" "20" "25")
hdim=("204" "162" "120" "108" "96")

for i in 0 1 2 3 4
do
  # ${hdim[i]} \
  # --repeat 3 \
  python main.py \
    --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.project lrgb-table \
    wandb.use True \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]}
  sleep 10
done