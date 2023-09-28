cd ../../
DATASET="peptides-struct_lg"
model="GCN"
# layer=("5" "8" "10" "12" "15" "20" "25" "30")
# hdim=("240" "198" "192" "174" "162" "144" "132" "120")
layer=("5" "10" "15" "20" "25")
hdim=("240" "186" "156" "138" "126")

for i in 0 1 2 3 4
do
  python main.py \
    --repeat 3 \
    --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.project lrgb-abl \
    wandb.use True \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.self_loop False
  sleep 10
done