DATASET="peptides-func"

for i in 3 5 7 9 11 13 15 17 19 21 23
do
  python main.py --cfg configs/GCNII/$DATASET-GCNII.yaml  wandb.use True  wandb.project lrgb  gnn.layers_mp $i
  sleep 10
done