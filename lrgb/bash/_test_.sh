
cd ../
DATASET="peptides-func"
model="GatedGCN"
layer=25
hdim=62

python main.py --cfg configs/$model/$DATASET-$model.yaml \
  wandb.use False \
  gnn.layers_mp $layer \
  gnn.dim_inner $hdim