cd ../../
DATASET="vocsuperpixels_lg"
model="GCN"
layer=("6" "8" "10" "12")
hdim=("252" "228" "204" "180")

i=2
for dropoutRate in 0.5 0.6
do
  python main.py --repeat 3 \
    --cfg configs/best/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.lgvariant 21 \
    gnn.dropout $dropoutRate
  sleep 10
done