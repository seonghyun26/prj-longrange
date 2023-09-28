cd ../../
DATASET="pcqm-contact_lg_bb"

python main.py --cfg configs/tuned/$DATASET-GCN.yaml \
  wandb.use True \
  wandb.project lrgb
  