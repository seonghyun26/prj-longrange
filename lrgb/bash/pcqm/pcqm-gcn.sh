cd ../../
DATASET="pcqm-contact_lg"

python main.py --cfg configs/tuned/$DATASET-GCN.yaml \
  wandb.use True \
  wandb.project lrgb \
  train.batch_size 4096
  