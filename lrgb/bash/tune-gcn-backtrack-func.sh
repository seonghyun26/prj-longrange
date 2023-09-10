cd ../
DATASET="peptides-func_lg_bt"
model="GCN"
layer=("10" "15" "20" "25")
hdim=("192" "162" "144" "132")

for dropoutrate in 0.0 0.1
do 
  for i in 3 2 1 0
  do
    # --repeat 3 \
    # ${hdim[i]} \
    python main.py \
      --cfg configs/tuned-backtrack/$DATASET-$model.yaml \
      wandb.project lrgb-table \
      wandb.use True \
      gnn.layers_mp ${layer[i]} \
      gnn.dim_inner 66 \
      gnn.residual False \
      gnn.dropout $dropoutrate \
      optim.schedule_patience 50
      # optim.max_epoch 300
    sleep 10
  done
done