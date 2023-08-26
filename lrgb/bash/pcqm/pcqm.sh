cd ../../
DATASET="pcqm-contact_lg"

python main.py --cfg configs/LG/pcqm/$DATASET-GCN.yaml \
  wandb.use False \
  gnn.lgvariant 20 \
  optim.max_epoch 400