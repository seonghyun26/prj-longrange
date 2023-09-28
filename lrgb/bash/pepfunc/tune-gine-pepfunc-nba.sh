cd ../../
DATASET="peptides-func_lg"
model="GINE"
# layer=("5" "7" "9" "10" "11" "12" "13" \
#   "15" "17" "19" "21" "23" "25" )
# hdim=("192" "168" "150" "144" "138" "126" "126" \
  # "120" "114" "108" "102" "96" "96" )
layer=("5" "10" "15" "20" "25")
hdim=("240" "186" "156" "138" "126")

for i in 0 1 2
do
    # --cfg configs/tuned/$DATASET-$model.yaml \
  python main.py \
    --repeat 3 \
    --cfg configs/best/$DATASET-$model.yaml \
    wandb.project lrgb-abl \
    wandb.use True \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]}
  sleep 10
done