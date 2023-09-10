cd ../../
DATASET="vocsuperpixels_lg"
model="GatedGCN"
layer=("6" "8" "10")
hdim=("120" "108" "96")

for i in 0 1 2
do
  python main.py --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} 
  sleep 10
done