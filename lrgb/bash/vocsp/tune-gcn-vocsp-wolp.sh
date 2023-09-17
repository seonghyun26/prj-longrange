cd ../../
DATASET="vocsuperpixels_lg"
model="GCN"
layer=("6" "8" "10" "12")
hdim=("252" "228" "204" "180")

for i in 2
do
  python main.py \
    --cfg configs/tuned/$DATASET-$model-wolp.yaml \
    --repeat 3 \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.lgvariant 21 
  sleep 10
done