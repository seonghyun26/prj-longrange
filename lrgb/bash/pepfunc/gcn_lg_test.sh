cd ../../
DATASET="peptides-func_lg"
model="GCN"

for layer in 15
do
  # python main.py \
  python main.py --repeat 3 \
    --cfg configs/LG/peptides-func/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 15 \
    gnn.dim_inner 164 \
    dataset.node_encoder_bn True \
    dataset.edge_encoder_bn True
  sleep 10
done