cd ../../
DATASET="peptides-func_lg"
model="GCN"
layer=("5" "15" "25")
hdim=("278" "162" "132")
length=${#layer[@]}
i=1

for encoderVersion in 2 3 4
do
  # python main.py \
  python main.py --repeat 3 \
    --cfg configs/LG/peptides-func/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    posenc_LapPE.version $encoderVersion
  sleep 10
done