cd ../../
DATASET="peptides-struct_lg"
model="GCN"

for dropoutrate in 0.0 0.1 0.2
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/LG/peptides-struct/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 25 \
    gnn.dim_inner 132 \
    gnn.dropout $dropoutrate
  sleep 10
done