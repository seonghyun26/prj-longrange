cd ../../
DATASET="peptides-struct_lg"
model="GCN"
layer=("5" "15" "25")
hdim=("278" "162" "132")
length=${#layer[@]}
dropoutrate=0.0

# for ((i=0;i<length;i++))
for i in 1
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/LG/peptides-struct/$DATASET-$model+LapPE.yaml \
    dataset.name peptides-structural_lg_vn \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate \
    posenc_LapPE.version 3
    # dataset.node_encoder_bn True \
    # dataset.edge_encoder_bn True \
  sleep 10
done