cd ../../
DATASET="peptides-struct_lg"
model="GatedGCN"
layer=("5" "15" "25")
hdim=("132" "78" "60")
length=${#layer[@]}
dropoutrate=0.0

for ((i=0;i<length;i++))
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/LG/peptides-struct/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate
  sleep 10
done