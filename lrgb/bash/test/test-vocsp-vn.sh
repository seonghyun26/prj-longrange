cd ../../
DATASET="vocsuperpixels_lg_vn"
model="GCN"
layer=("5" "10" "15")
hdim=("132" "94" "72")
length=${#layer[@]}
dropoutrate=0.0

for i in 0 1 2
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/LG_VN/vocsp/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate \
    posenc_LapPE.version 3
  sleep 10
done