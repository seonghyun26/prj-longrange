cd ../../
DATASET="peptides-func_lg_bb"
model="GCN"

# python main.py \
#   --repeat 3 \
#   --cfg configs/best/$DATASET-$model.yaml \
#   wandb.project lrgb-table \
#   wandb.use True \
#   gnn.self_loop True

for self_loop in True False
do 
  for batchsize in 128 200 256
  do
    python main.py \
      --cfg configs/best/$DATASET-$model.yaml \
      wandb.project lrgb-table \
      wandb.use True \
      gnn.self_loop $self_loop \
      train.batch_size $batchsize
  done
done