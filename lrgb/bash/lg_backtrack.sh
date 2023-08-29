cd ../

DATASET="peptides-func_lg_backtrack"
model="GCN"
layer=("5" "15" "25")
hdim=("278" "162" "132")
length=${#layer[@]}

for ((i=0;i<length;i++))
do
  python main.py --repeat 3 \
    --cfg configs/LG_backtrack/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]}
  sleep 10
done