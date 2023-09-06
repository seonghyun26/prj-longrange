cd ../../
DATASET1="peptides-struct_lg"
model="GCN"
layer=("5" "10" "15" "25")
hdim=("264" "204" "162" "132")
length=${#layer[@]}
dropoutrate=0.0
encoderVersion=3
i=0

for weight_decay in 0.00001
#  0.1
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/LG/peptides-struct/$DATASET1-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate \
    posenc_LapPE.version $encoderVersion \
    optim.weight_decay $weight_decay
  sleep 10
done
