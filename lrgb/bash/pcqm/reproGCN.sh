cd ../../
DATASET="pcqm-contact"
model="GCN"
layer=("5" "10" "15")
hdim=("278" "168" "132")
length=${#layer[@]}

for i in 0
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/$model/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    train.batch_size 2048
    # gnn.layers_mp ${layer[i]} \
    # gnn.dim_inner ${hdim[i]} \
  sleep 10
done