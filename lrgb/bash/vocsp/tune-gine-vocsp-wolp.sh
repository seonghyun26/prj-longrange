cd ../../
DATASET="vocsuperpixels_lg"
model="GINE"
layer=("6" "8" "10")
hdim=("192" "168" "150")

for i in 2
do
  python main.py \
    --cfg configs/tuned/$DATASET-$model-wolp.yaml \
    --repeat 3 \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} 
  sleep 10
done