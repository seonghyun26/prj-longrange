cd ../../
DATASET="vocsuperpixels_lg"
model="GCN"
layer=("5" "10" "15")
hdim=("258" "200" "162")
length=${#layer[@]}
dropoutrate=0.15

for i in 0 1 2
do
  python main.py --cfg configs/LG/vocsp/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate
  sleep 10
done