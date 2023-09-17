cd ../../
DATASET="vocsuperpixels"
model="GCN"
layer=("6" "8" "10" "12")
hdim=("252" "228" "204" "180")

i=2
for dropoutRate in 0.2 0.4 0.6
do
  python main.py \
    --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.lgvariant 21 \
    gnn.dropout $dropoutRate
  sleep 10
done

# Vanilla GCN with dropout