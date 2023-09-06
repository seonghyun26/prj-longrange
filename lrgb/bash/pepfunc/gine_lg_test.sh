cd ../../
DATASET="peptides-func_lg"
model="GINE"
layer=("5" "15" "25")
hdim=("268" "162" "132")
length=${#layer[@]}
i=0
dropoutrate=0.0
encoderVersion=6

for i in 0 1 2
do
  python main.py --repeat 3 \
    --cfg configs/LG/peptides-func/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate \
    posenc_LapPE.version $encoderVersion
  sleep 10
done