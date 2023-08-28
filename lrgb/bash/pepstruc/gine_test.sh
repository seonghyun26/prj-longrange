cd ../../
DATASET="peptides-struct"
model="GINE"

for layer in 15
do
  # python main.py \
  python main.py --repeat 3 \
    --cfg configs/$model/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 15 \
    gnn.dim_inner 122 \
    dataset.node_encoder_bn True \
    dataset.edge_encoder_bn True
  sleep 10
done

for layer in 15
do
  # python main.py \
  python main.py --repeat 3 \
    --cfg configs/$model/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 15 \
    gnn.dim_inner 122 \
    dataset.node_encoder_bn False \
    dataset.edge_encoder_bn False
  sleep 10
done