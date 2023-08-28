cd ../../

DATASET="peptides-func_lg"
model="GCN"
layer=("5" "15" "25")
hdim=("278" "170" "132")
length=${#layer[@]}

for ((i=0;i<length;i++))
do
  for dropoutrate in 0.0 0.01 0.1
  do
    python main.py --cfg configs/LG/peptides-func/$DATASET-$model+LapPE.yaml \
      wandb.use True \
      wandb.project lrgb \
      gnn.layers_mp ${layer[i]} \
      gnn.dim_inner ${hdim[i]} \
      gnn.dropout $dropoutrate
    sleep 10
  done
done