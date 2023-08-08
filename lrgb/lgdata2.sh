DATASET="peptides-funclg"
model=$1
for i in 5 10 15 17 20 25
do
  python main.py --cfg configs/LG/$DATASET-$model.yaml  wandb.use True  wandb.project lrgb  gnn.layers_mp $i
  sleep 10
done