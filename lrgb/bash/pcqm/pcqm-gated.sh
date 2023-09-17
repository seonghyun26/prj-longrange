cd ../../
DATASET="pcqm-contact_lg"

python main.py --cfg configs/tuned/$DATASET-GatedGCN.yaml \
  wandb.use True \
  wandb.project lrgb-table
  