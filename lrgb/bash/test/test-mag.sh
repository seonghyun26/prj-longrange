cd ../../
DATASET="peptides-struct_lg"
model="GCN"
layer=("5" "15" "25")
hdim=("264" "162" "132")
length=${#layer[@]}
dropoutrate=0.01

# for ((i=0;i<length;i++))
# for dropoutrate in 0.0 0.01 0.015
for i in 0 1 2
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/LG/peptides-struct/$DATASET-$model+MagLapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate \
    posenc_LapPE.version 3
  sleep 10
done