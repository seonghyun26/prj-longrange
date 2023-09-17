cd ../../
DATASET="peptides-func_lg_bb"
model="GCN"
layer=("5" "8" "10" "12" "14""15" "20" "25" "30" "35")
hdim=("240" "204" "192" "174" "168" "162" "144" "132" "120" "108")


# dropout=0.1 0.15
for i in 3 4
  do
  for dropout in 0.1 0.08 0.05 
    do
      # --repeat 3 \
    python main.py \
      --cfg configs/tuned/$DATASET-$model.yaml \
      wandb.project lrgb-table \
      wandb.use True \
      gnn.layers_mp ${layer[i]} \
      gnn.dim_inner ${hdim[i]} \
      gnn.dropout $dropout
    sleep 10
  done
done