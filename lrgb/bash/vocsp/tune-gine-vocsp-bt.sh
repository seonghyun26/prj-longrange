cd ../../
DATASET="vocsuperpixels_lg_bt"
model="GINE"
layer=("6" "8" "10" "12")
hdim=("192" "168" "150" "138")

for i in 0 1 2 3
do
  python main.py \
    --repeat 3 \
    --cfg configs/tuned-backtrack/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} 
  sleep 10
done