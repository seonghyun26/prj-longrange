cd ../../
DATASET="vocsuperpixels"
layer=("5" "10" "15")
hdim=("240" "200" "160")
length=${#layer[@]}

for ((i=0;i<length;i++))
do
  python main.py --repeat 3 \
    --cfg configs/GCNII/$DATASET-GCNII+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]}
  sleep 10
done