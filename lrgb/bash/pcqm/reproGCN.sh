cd ../../
DATASET="pcqm-contact"
model="GCN"
layer=("5" "10" "15")
hdim=("278" "168" "132")
length=${#layer[@]}

for ((i=0;i<length;i++))
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/$model/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    train.batch_size 1024
  sleep 10
done