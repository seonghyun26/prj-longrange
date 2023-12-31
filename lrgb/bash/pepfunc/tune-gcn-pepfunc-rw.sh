cd ../../
DATASET="peptides-func_lg"
model="GCN"
layer=("5" "10" "15" "20" "25")
hdim=("252" "192" "162" "144" "132")

for i in 0 1 2 3 4
do
  # ${hdim[i]} \
    # --repeat 3 \
  python main.py \
    --cfg configs/tuned/$DATASET-$model+RWSE.yaml \
    wandb.project lrgb-table \
    wandb.use True \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]}
  sleep 10
done