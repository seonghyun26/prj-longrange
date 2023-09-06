cd ../../
DATASET="pcqm-contact_lg"

python main.py --cfg configs/LG/pcqm/$DATASET-GCN+LapPE.yaml \
  wandb.use True \
  wandb.project lrgb \
  optim.max_epoch 300 \
  train.batch_size 1024
  