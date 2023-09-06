cd ../../
DATASET="peptides-struct_lg"
model="GCN"
layer=("5" "15" "25")
hdim=("270" "162" "132")
length=${#layer[@]}
i=0
# encoderVersion=2
dropoutrate=0.0

for encoderVersion in 2 3 4 5 6 7
do
  # python main.py --repeat 2 \
  python main.py \
    --cfg configs/LG/peptides-struct/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate \
    posenc_LapPE.version $encoderVersion
  sleep 10
done