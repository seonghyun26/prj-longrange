cd ../../
DATASET="peptides-struct_lg"
model="GatedGCN"
# layer=("5" "15" "25")
# hdim=("134" "78" "60")
layer=15
hdim=78
length=${#layer[@]}

for ((i=0;i<length;i++))
do
  for dropoutrate in 0.01
  do
    # python main.py
    python main.py --repeat 3 \
      --cfg configs/LG/peptides-struct/$DATASET-$model+LapPE.yaml \
      wandb.use True \
      wandb.project lrgb \
      gnn.layers_mp ${layer[i]} \
      gnn.dim_inner ${hdim[i]} \
      gnn.dropout $dropoutrate
    sleep 10
  done
done