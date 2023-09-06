cd ../../
DATASET="peptides-func_lg"
model="GCN"
layer=("5" "15" "25")
hdim=("278" "162" "132")
length=${#layer[@]}
dropoutrate=0.0
i=1

# for ((i=0;i<length;i++))
for encoderVersion in 2 3 4 5 6
do
  # python main.py \
  python main.py --repeat 2 \
    --cfg configs/LG/peptides-func/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate \
    posenc_LapPE.version $encoderVersion
  sleep 10
done
