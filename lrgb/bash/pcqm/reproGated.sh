cd ../../
DATASET="pcqm-contact"
model="GatedGCN"
layer=("5" "10" "15")
hdim=("136" "78" "60")
length=${#layer[@]}

for ((i=0;i<length;i++))
do
  python main.py --repeat 3 \
    --cfg configs/$model/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]}
  sleep 10
done