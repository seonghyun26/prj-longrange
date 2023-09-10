cd ../../
DATASET="vocsuperpixels_lg_bt"
model="GINE"
layer=("6" "8" "10")
hdim=("192" "168" "150")

for i in 0 1 2
do
  python main.py --cfg configs/tuned-backtrack/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} 
  sleep 10
done