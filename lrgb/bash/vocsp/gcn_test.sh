cd ../../
DATASET="vocsuperpixels"
model="GCN"
for layer in 15
do
  python main.py --cfg configs/$model/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 7 \
    gnn.dim_inner 258
  sleep 10
done