cd ../../
DATASET="peptides-func_lg_bt"
model="GCN"
layer=("5" "10" "15" "20" "25" "30")
hdim=("240" "192" "162" "144" "132" "120")

for i in 2 3 4 5 
do
  # ${hdim[i]} \
  python main.py \
    --repeat 3 \
    --cfg configs/tuned-backtrack/$DATASET-$model.yaml \
    wandb.project lrgb-table \
    wandb.use True \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]}
  sleep 10
done