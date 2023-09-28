cd ../../
DATASET="vocsuperpixels_lg"
model="GCN"
layer=("6" "8" "10" "12")
hdim=("252" "222" "198" "180")

for i in 3
do
  for dropoutRate in 0.7
  do
    python main.py \
      --repeat 3 \
      --cfg configs/best/$DATASET-$model.yaml \
      wandb.use True \
      wandb.project lrgb-table \
      gnn.layers_mp ${layer[i]} \
      gnn.dim_inner ${hdim[i]} \
      gnn.dropout $dropoutRate
    sleep 10
  done
done