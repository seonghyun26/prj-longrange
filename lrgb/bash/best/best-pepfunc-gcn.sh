cd ../../
DATASET="peptides-func_lg_bb"
model="GCN"

  # --repeat 3 \
python main.py \
  --cfg configs/best/$DATASET-$model.yaml \
  wandb.project lrgb-table \
  wandb.use True