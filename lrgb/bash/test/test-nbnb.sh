cd ../../
DATASET="peptides-struct_lg_nbnb"
model="GCN"
layer=("5" "15" "25")
hdim=("267" "162" "132")
length=${#layer[@]}
i=0
dropoutrate=0.012

for lgvariant in 30
do
  # python main.py --repeat 2 \
  python main.py \
    --cfg configs/LG_NBNB/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate \
    gnn.lgvariant 30
  sleep 10
done