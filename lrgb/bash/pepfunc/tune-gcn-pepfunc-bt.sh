cd ../../
DATASET="peptides-func_lg_bt"
model="GCN"
layer=("5" "10" "15" "20" "25")
hdim=("240" "186" "156" "138" "126")

for i in 2
do
  python main.py \
    --repeat 3 \
    --cfg configs/tuned-backtrack/$DATASET-$model.yaml \
    wandb.project lrgb-abl \
    wandb.use True \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.self_loop False
  sleep 10
done