cd ../../
DATASET="pcqm-contact_lg"
layer=("5" "10" "15")
hdim=("220" "200" "168")
length=${#layer[@]}

for ((i=0;i<length;i++))
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/LG/pcqm/$DATASET-GCN+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    train.batch_size 4096 \
    optim.max_epoch 400
  sleep 10
done