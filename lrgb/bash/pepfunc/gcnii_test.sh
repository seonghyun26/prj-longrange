cd ../../
DATASET="peptides-func"
model="GCNII"

for layer in 15
do
  python main.py --repeat 3 \
    --cfg configs/$model/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp 15 \
    gnn.dim_inner 176
  sleep 10
done