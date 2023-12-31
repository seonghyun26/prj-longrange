cd ../../
DATASET="peptides-func_lg_bb"
model="GINE"
# layer=("5" "7" "9" "10" "11" "12" "13" \
#   "15" "17" "19" "21" "23" "25" )
# hdim=("192" "168" "150" "144" "138" "126" "126" \
  # "120" "114" "108" "102" "96" "96" )
layer=("6" "8" "10" "12" "20")
hdim=("192" "150" "144" "132" "102")

for i in 2
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