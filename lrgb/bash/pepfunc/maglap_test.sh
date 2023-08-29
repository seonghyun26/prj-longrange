cd ../../
DATASET="peptides-func_lg"
model="GCN"

for dropoutrate in 0.0
do
  # python main.py \
  python main.py --repeat 3 \
    --cfg configs/LG/peptides-func/$DATASET-$model+MagLapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 15 \
    gnn.dim_inner 162 \
    gnn.dropout $dropoutrate 
  sleep 10
done