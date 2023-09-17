cd ../../
DATASET="peptides-func_lg_bb"
model="GCN"
layer=("5" "10" "15" "20" "25" "30" "35")
hdim=("240" "192" "162" "144" "132" "120" "108")

for i in 3
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