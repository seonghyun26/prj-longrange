cd ../../
DATASET="peptides-struct_lg_bb"
model="GatedGCN"
layer=("6" "8" "10" "12")
hdim=("96" "102" "96" "84")

for i in 1 2
do
  python main.py \
    --repeat 3 \
    --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]}
  sleep 10
done