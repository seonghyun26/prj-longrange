cd ../../
DATASET="peptides-func"
model="GCN"
tags=("NodeEncode_BN" "EdgeEncode_BN")
tags_list=$(IFS=,; echo "${tag[*]}")

for layer in 15
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/$model/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 15 \
    gnn.dim_inner 172 \
    dataset.node_encoder_bn True \
    dataset.edge_encoder_bn True
  sleep 10
done