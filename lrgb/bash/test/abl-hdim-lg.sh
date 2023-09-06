cd ../../
DATASET="peptides-func_lg"
model="GCN"
layer=15
# hdim=("258" "126" "66" "30")
hdim=("516" "1026")
length=${#layer[@]}
dropoutrate=0.0

for i in 0 1
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/LG/peptides-func/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $layer \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate \
    posenc_LapPE.version 3
  sleep 10
done