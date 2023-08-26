cd ../../
DATASET="peptides-func"
model="GatedGCN"
for layer in 15
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/$model/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 15 \
    gnn.dim_inner 78
  sleep 10
done