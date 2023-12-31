cd ../../
DATASET="vocsuperpixels"
model="GatedGCN"
for layer in 15
do
  python main.py --cfg configs/$model/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 10 \
    gnn.dim_inner 96 \
    dataset.node_encoder_bn True \
    dataset.edge_encoder_bn True \
    train.batch_size 16
  sleep 10
done