cd ../
DATASET="peptides-struct_lg"
model="GCN"
layer=("10" "15" "20" "25")
hdim=("192" "162" "144" "132")

for i in 0 1 2 3
do
  for dropoutrate in 0.1 0.0
  do
      # --repeat 3 \
    python main.py \
      --cfg configs/tuned/$DATASET-$model.yaml \
      wandb.project lrgb-table \
      wandb.use True \
      gnn.layers_mp ${layer[i]} \
      gnn.dim_inner ${hdim[i]} \
      gnn.dropout $dropoutrate \
      optim.schedule_patience 20
      # gnn.residual "False" \
      # optim.max_epoch 300
    sleep 10
  done
done