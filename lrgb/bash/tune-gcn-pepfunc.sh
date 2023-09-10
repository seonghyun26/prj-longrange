cd ../
DATASET="peptides-func_lg"
model="GCN"
layer=("10" "15" "20" "25")
hdim=("192" "162" "144" "132")

for i in 0 1 2 3
do
  # --repeat 3 \
  # ${hdim[i]} \
  python main.py \
    --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.project lrgb-table \
    wandb.use True \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]}
  sleep 10
done