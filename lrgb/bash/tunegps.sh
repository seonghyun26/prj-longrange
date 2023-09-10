cd ../

DATASET=$1
# peptides-func
# peptides-struct
# vocsuperpixels
# --repeat 3 \

python main.py \
  --cfg configs/tuned/$DATASET-GPS.yaml \
  wandb.project lrgb-table \
  wandb.use True \
  optim.max_epoch 400 \
  train.batch_size 50
