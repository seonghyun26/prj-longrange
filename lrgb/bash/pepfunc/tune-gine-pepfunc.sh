cd ../../

# DATASET=$1
# peptides-func
# peptides-struct
# vocsuperpixels
model="GINE"
for DATASET in "peptides-func"
do
  python main.py \
    --repeat 3 \
    --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.project lrgb-table \
    wandb.use True \
    gnn.residual False
  sleep 10
done