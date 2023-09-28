cd ../../
DATASET1="cifar10"
model="GCN"
# layer=("5" "15" "25")
# hdim=("276" "162" "132")
layer=("2" "4" "6" "8" "10" "12")
hdim=("252" "282" "252" "222" "198" "180")

dropout=0.2
for i in 0 2
do
  # --repeat 3 \
  python main.py \
    --cfg configs/GCN/$DATASET1-$model.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.self_loop False \
    gnn.dropout $dropout \
    train.batch_size 200
  sleep 10
done