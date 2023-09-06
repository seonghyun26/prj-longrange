cd ../../
DATASET="peptides-struct"
model="GCNII"
layer=("5" "15" "25")
hdim=("268" "162" "132")
length=${#layer[@]}
dropoutrate=0.0

for ((i=0;i<length;i++))
do
  # python main.py \
  python main.py --repeat 3 \
    --cfg configs/$model/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    posenc_LapPE.version 3
  sleep 10
done