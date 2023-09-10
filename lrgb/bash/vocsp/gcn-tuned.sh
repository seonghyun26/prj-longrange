cd ../../
DATASET="vocsuperpixels_lg"
model="GCN"
layer=("5" "10" "15")
hdim=("258" "204" "162")
length=${#layer[@]}

for i in 0 1 2
do
  python main.py --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} 
  sleep 10
done