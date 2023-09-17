cd ../../
DATASET="pcqm-contact_lg"

python main.py --cfg configs/tuned/$DATASET-GINE.yaml \
  wandb.use True \
  wandb.project lrgb-table \
  train.batch_size 4096
  