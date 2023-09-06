cd ../../
DATASET1="peptides-func_lg"
DATASET2="peptides-struct_lg"
model="GatedGCN"
layer=("5" "15" "25")
hdim=("132" "78" "60")
length=${#layer[@]}
dropoutrate=0.0
encoderVersion=3

# for ((i=0;i<length;i++))
for i in 1
do
  # python main.py \
  python main.py --repeat 3 \
    --cfg configs/LG/peptides-func/$DATASET1-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate \
    posenc_LapPE.version $encoderVersion \
    gnn.lgvariant 13
  sleep 10

  # python main.py \
  python main.py --repeat 3 \
    --cfg configs/LG/peptides-struct/$DATASET2-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate \
    posenc_LapPE.version $encoderVersion \
    gnn.lgvariant 13
  sleep 10
done
