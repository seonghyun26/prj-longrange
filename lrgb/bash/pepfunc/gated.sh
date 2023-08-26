cd ../../
DATASET="peptides-func"
layer=("5" "15" "25")
hdim=("132" "78" "60")
length=${#layer[@]}

for ((i=0;i<length;i++))
do
  python main.py --repeat 3 \
    --cfg configs/GatedGCN/$DATASET-GatedGCN+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.residual False \
    gnn.lgvariant 12 \
    optim.max_epoch 500 
done
