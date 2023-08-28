cd ../../
DATASET="peptides-func_lg"
model="GCN"
layer=("5" "15" "25")
hdim=("278" "162" "132")
length=${#layer[@]}
dropoutrate=0.01

for ((i=0;i<length;i++))
do
  # python main.py \
  python main.py --repeat 3 \
    --cfg configs/LG/peptides-func/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate 
    # dataset.node_encoder_bn True \
    # dataset.edge_encoder_bn True \
  sleep 10
done