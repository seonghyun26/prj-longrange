cd ../../
DATASET="peptides-func_lg_bb"
model="GatedGCN"
# layer=("5" "10" "15" "20" "25")
# hdim=("132" "96" "78" "66" "60")
layer=("6" "8" "10" "12")
hdim=("120" "108" "96" "90")

for i in 2 3
do
  python main.py \
    --repeat 3 \
    --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    train.batch_size 100 \
    gnn.residual False
  sleep 10
done

for i in 2 3
do
  python main.py \
    --repeat 3 \
    --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.use True \
    wandb.project lrgb-table \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    train.batch_size 200 \
    gnn.residual False
  sleep 10
done