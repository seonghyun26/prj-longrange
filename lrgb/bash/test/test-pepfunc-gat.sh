cd ../../
DATASET1="peptides-func_lg"
model="GAT"
layer=("5" "15" "25")
hdim=("276" "162" "132")
length=${#layer[@]}
dropoutrate=0.0
encoderVersion=3

for i in 0 1 2
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/test/$DATASET1-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp ${layer[i]} \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate \
    posenc_LapPE.version $encoderVersion
  sleep 10
done
