cd ../../
DATASET="peptides-func"
model="GCN"
# layer=("5" "7" "9" "11" "13" \
# "15" "17" "19" "21" "23" "25")
# hdim=("240" "216" "192" "180" "168" \
# "156" "150" "144" "138" "132" "126")
layer=("5" "10" "15" "20" "25")
hdim=("240" "186" "156" "138" "126")


# dropout=0.1 0.15
for i in 0 1 2 3
do
  python main.py \
    --repeat 3 \
    --cfg configs/$model/$DATASET-$model+LapPEorg.yaml \
    wandb.project lrgb-abl \
    wandb.use True \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    optim.max_epoch 300
  sleep 10
done