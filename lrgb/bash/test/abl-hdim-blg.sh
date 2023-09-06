cd ../../
DATASET="peptides-func_lg_backtrack"
model="GCN"
layer=15
hdim=("66" "126" "258" "516" "1026")
length=${#layer[@]}
dropoutrate=0.0

for i in 0 1 2 3 4
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/LG_backtrack/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $layer \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate \
    posenc_LapPE.version 3
  sleep 10
done