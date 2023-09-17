cd ../../
DATASET="peptides-func_lg_bb"
model="GINE"
layer=("5" "7" "9" "11" "13" \
  "15" "17" "19" "21" "23" "25" )
hdim=("192" "168" "150" "138" "126" \
  "120" "114" "108" "102" "96" "96" )

for i in 0 1 2 3 4 5 6 7 8 9 10
do
  # ${hdim[i]} \
    # --repeat 3 \
  python main.py \
    --cfg configs/tuned/$DATASET-$model.yaml \
    wandb.project lrgb-table \
    wandb.use True \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]}
  sleep 10
done