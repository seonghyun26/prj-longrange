cd ../

DATASET=$1
# peptides-func
# peptides-struct
# vocsuperpixels
for model in "GCN" "GINE" "GatedGCN"
do
    # --repeat 3 \
  python main.py \
    --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.project lrgb-table \
    wandb.use True
  sleep 10
done