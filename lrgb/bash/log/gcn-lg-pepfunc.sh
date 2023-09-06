cd ../../
DATASET="peptides-func_lg"
model="GCN"
layer=("10" "15" "20")
hdim=("210" "162" "150")
length=${#layer[@]}
dropoutrate=0.0

for i in 0 1 2
do
  # python main.py \
  python main.py --repeat 3 \
    --cfg configs/LG/peptides-func/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    posenc_LapPE.version 2
  sleep 10
done