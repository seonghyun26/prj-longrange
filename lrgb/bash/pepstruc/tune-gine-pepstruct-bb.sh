cd ../../
DATASET="peptides-struct_lg_bb"
model="GINE"
layer=("6" "8" "10" "12")
hdim=("192" "156" "144" "132")

for i in 0 1 2 3
do
  python main.py --cfg configs/tuned/$DATASET-$model.yaml \
    --repeat 3 \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} 
  sleep 10
done