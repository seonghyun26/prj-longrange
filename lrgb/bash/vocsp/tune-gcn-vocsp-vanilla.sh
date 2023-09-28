cd ../../
DATASET="vocsuperpixels"
model="GCN"
layer=("6" "8" "10" "12")
hdim=("252" "222" "198" "180")

dropoutRate=0.7
for i in 0 1 2 3
do
  python main.py \
    --repeat 3 \
    --cfg configs/$model/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    optim.max_epoch 150
  sleep 10
done