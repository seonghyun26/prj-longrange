cd ../../
DATASET="vocsuperpixels_lg"
model="GINE"
layer=("5" "10" "15")
hdim=("204" "144" "120")
length=${#layer[@]}
dropoutrate=0.15

for i in 2
do
  # python main.py \
  python main.py --repeat 3 \
    --cfg configs/LG/vocsp/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate
  sleep 10
done