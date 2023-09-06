cd ../../
DATASET="peptides-func_lg_vn"
model="GatedGCN"
layer=("5" "15" "25")
hdim=("132" "78" "60")
length=${#layer[@]}
dropoutrate=0.0


# for ((i=0;i<length;i++))
for i in 0 1 2
do
  # python main.py \
  python main.py --repeat 3 \
    --cfg configs/LG_VN/peptides-func/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate \
    posenc_LapPE.version 3
  sleep 10
done