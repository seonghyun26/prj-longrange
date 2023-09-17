cd ../../
DATASET="vocsuperpixels_lg_bt"
model="GatedGCN"
layer=("6" "8" "10")
hdim=("120" "108" "90")

for i in 2 1 0
do
  python main.py --cfg configs/tuned-backtrack/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} 
  sleep 10
done