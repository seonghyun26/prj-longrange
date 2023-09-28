cd ../../
DATASET1="cifar10_lg"
model="GCN"
# layer=("5" "15" "25")
# hdim=("276" "162" "132")
layer=("4" "6" "8" "10" "12")
hdim=("282" "252" "222" "198" "180")

dropout=0.2
for i in 1
do
  # --repeat 3 \
  python main.py \
    --cfg configs/GCN/$DATASET1-$model.yaml \
    wandb.use True \
    wandb.project lrgb-abl \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropout \
    train.batch_size 64
  sleep 10
done
