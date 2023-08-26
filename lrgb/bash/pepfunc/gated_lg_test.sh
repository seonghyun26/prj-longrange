cd ../../
DATASET="peptides-func_lg"
model="GatedGCN"
for layer in 15
do
  # python main.py \
  python main.py --repeat 3 \
    --cfg configs/LG/peptides-func/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 15 \
    gnn.dim_inner 78
  sleep 10
done