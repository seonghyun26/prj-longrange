cd ../../
DATASET="vocsuperpixels_lg"
model="GatedGCN"
layer=("5" "10" "15")
hdim=("132" "94" "72")
length=${#layer[@]}
dropoutrate=0.15

for ((i=0;i<length;i++))
do
  # python main.py \
  python main.py --repeat 3 \
    --cfg configs/LG/vocsp/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 10 \
    gnn.dim_inner 94 \
    gnn.residual False \
    gnn.batchnorm True \
    gnn.dropout $dropoutrate \
    optim.max_epoch 300
  sleep 10
done