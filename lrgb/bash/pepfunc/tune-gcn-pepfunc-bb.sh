cd ../../
DATASET="peptides-func_lg_bb"
model="GCN"
# layer=("5" "7" "9" "11" "13" \
# "15" "17" "19" "21" "23" "25")
# hdim=("240" "216" "192" "180" "168" \
# "156" "150" "144" "138" "132" "126")
layer=("5" "9" "10" "11" "15" "20" "25")
hdim=("240" "192" "186" "180" "156" "138" "126")


# dropout=0.1 0.15
for i in 0
do
  python main.py \
    --repeat 3 \
    --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.project lrgb-abl \
    wandb.use True \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.self_loop False \
    train.batch_size 128
  sleep 10
done